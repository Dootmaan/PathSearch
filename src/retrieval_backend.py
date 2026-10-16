"""Shared retrieval backend: Bag-of-Barcodes (BoB) class and distance utilities.

This module consolidates the BoB representation and all pairwise-distance
logic that was previously duplicated across every test script.  Importing
from here ensures consistent behaviour and makes it easier to adjust
retrieval hyper-parameters in a single place.

Public API
----------
BoB
    Data class holding per-slide barcodes and a semantic embedding.
build_bobs(case_ids, mosaics, img_feats, labels, semantic_weight)
    Construct a {case_id -> BoB} dict from encoding outputs.
compute_all_distances(bobs)
    O(N² · M²) Python-loop baseline (small archives, easy to reason about).
compute_all_distances_batched(bobs, chunk)
    Vectorised variant using scipy cdist + numpy broadcasting.  Avoids nested
    Python iteration over barcodes; suitable for archives of hundreds of slides.

The ``semantic_weight`` parameter (default 200.0) controls the relative
contribution of the L2 semantic term.  Pass it explicitly rather than
embedding a magic number in each test script.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import bitarray
import numpy as np
import torch
from bitarray import util as butil

__all__ = [
    "BoB",
    "build_bobs",
    "compute_all_distances",
    "compute_all_distances_batched",
]

# Default weight for the semantic L2 term in BoB distance.
DEFAULT_SEMANTIC_WEIGHT: float = 200.0


# ---------------------------------------------------------------------------
# BoB class
# ---------------------------------------------------------------------------

class BoB:
    """Bag-of-Barcodes representation for a single slide.

    Args:
        barcodes: Per-patch binary barcodes, one ``bitarray`` per patch.
        semantic_embedding: Slide-level semantic feature tensor, shape (1, D)
            or (D,).  Must be on CPU.
        name: Slide / case identifier string.
        site: Integer class label (or any comparable object).
        semantic_weight: Weight applied to the L2 semantic distance term when
            calling :meth:`distance`.  Defaults to
            :data:`DEFAULT_SEMANTIC_WEIGHT`.
    """

    def __init__(
        self,
        barcodes: List[bitarray.bitarray],
        semantic_embedding: torch.Tensor,
        name: str,
        site,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> None:
        self.barcodes = barcodes
        self.semantic_embedding = semantic_embedding.cpu()
        self.name = name
        self.site = site
        self.semantic_weight = semantic_weight

    def distance(self, other: "BoB") -> float:
        """Asymmetric BoB distance between *self* (query) and *other* (gallery).

        Distance = median_{f in self.barcodes}( min_{g in other.barcodes} hamming(f, g) )
                 + semantic_weight * ||self.sem - other.sem||_2

        The Hamming term is derived from the XOR bit-count (``bitarray.util.count_xor``).
        """
        min_hamming_per_query_bc: List[int] = []
        for feat in self.barcodes:
            xor_counts = [butil.count_xor(feat, b) for b in other.barcodes]
            min_hamming_per_query_bc.append(int(np.min(xor_counts)))
        hamming_part = float(np.median(min_hamming_per_query_bc))
        sem_part = float(
            torch.norm(self.semantic_embedding - other.semantic_embedding, p=2).item()
        )
        return hamming_part + self.semantic_weight * sem_part


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------

def build_bobs(
    case_ids: List[str],
    mosaics: List[torch.Tensor],
    img_feats: List[torch.Tensor],
    labels: List,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> Dict[str, BoB]:
    """Construct a BoB dict from slide encoding outputs.

    Args:
        case_ids: Slide identifiers, one per slide.
        mosaics: Concatenated mosaic+image tensors, one per slide.  Each is
            shape (num_patches, D).
        img_feats: Slide-level semantic embeddings, shape (1, D) each.
        labels: Integer class labels, one per slide.
        semantic_weight: Forwarded to each :class:`BoB`.

    Returns:
        Dict mapping ``case_id -> BoB``.
    """
    bobs: Dict[str, BoB] = {}
    for i, case_id in enumerate(case_ids):
        mosaic_np = np.array(mosaics[i])  # (num_patches, D)
        # Barcodes: +1 where the feature decreases across dimensions (diff < 0).
        barcodes_np = (np.diff(mosaic_np, n=1, axis=1) < 0).astype(int)  # (P, D-1)
        bc_list = [bitarray.bitarray(row.tolist()) for row in barcodes_np]

        label = labels[i]
        if isinstance(label, torch.Tensor):
            label = label.item() if label.numel() == 1 else label

        bobs[case_id] = BoB(
            barcodes=bc_list,
            semantic_embedding=img_feats[i].reshape(1, -1),
            name=case_id,
            site=label,
            semantic_weight=semantic_weight,
        )
    return bobs


# ---------------------------------------------------------------------------
# Distance computation – baseline Python loop
# ---------------------------------------------------------------------------

def compute_all_distances(
    bobs: Dict[str, BoB],
) -> Dict[str, List[Tuple[float, str]]]:
    """Compute all-pairs BoB distances using Python loops.

    Suitable for small benchmark sets (up to ~100 slides).  For larger
    archives prefer :func:`compute_all_distances_batched`.

    Returns:
        Dict mapping ``case_id -> [(dist, peer_case_id), ...]`` (unsorted).
    """
    distances: Dict[str, List] = defaultdict(list)
    keys = list(bobs.keys())
    for a in keys:
        for b in keys:
            if a == b:
                continue
            distances[a].append((bobs[a].distance(bobs[b]), b))
    return distances


# ---------------------------------------------------------------------------
# Distance computation – vectorised (scipy + numpy)
# ---------------------------------------------------------------------------

def compute_all_distances_batched(
    bobs: Dict[str, BoB],
    chunk: int = 2048,
) -> Dict[str, List[Tuple[float, str]]]:
    """Compute all-pairs BoB distances using numpy/scipy batch operations.

    The Hamming term is computed via ``scipy.spatial.distance.cdist`` on the
    flattened (N·M, L) barcode matrix, avoiding the O(N² · M²) Python loop.
    The semantic L2 term is computed via ``torch.cdist`` on stacked embeddings.

    Memory usage is O((N·M)²) float32 for the Hamming matrix.  For very large
    archives pass a smaller ``chunk`` to tile the computation.

    Args:
        bobs: BoB dict as returned by :func:`build_bobs`.
        chunk: Row batch size for the cdist computation.  Reduce if running
            out of memory.

    Returns:
        Dict mapping ``case_id -> [(dist, peer_case_id), ...]`` (unsorted).
        Falls back to :func:`compute_all_distances` if scipy is unavailable.
    """
    try:
        from scipy.spatial.distance import cdist as scipy_cdist
    except ImportError:
        logging.warning(
            "scipy unavailable; falling back to Python-loop BoB distances."
        )
        return compute_all_distances(bobs)

    keys = list(bobs.keys())
    N = len(keys)
    if N == 0:
        return defaultdict(list)

    semantic_weight = bobs[keys[0]].semantic_weight

    # ---- Semantic distances (N, D) → (N, N) -----------------------------------
    sem_feats = torch.cat(
        [bobs[k].semantic_embedding.reshape(1, -1) for k in keys], dim=0
    )  # (N, D)
    sem_dist_mat = torch.cdist(sem_feats.float(), sem_feats.float()).numpy()  # (N, N)

    # ---- Barcode Hamming distances via scipy cdist ----------------------------
    M = len(bobs[keys[0]].barcodes)
    L = len(bobs[keys[0]].barcodes[0]) if M > 0 else 1

    # Convert barcodes to float32 numpy (N, M, L), flatten to (N*M, L).
    all_bc = np.zeros((N, M, L), dtype=np.float32)
    for i, k in enumerate(keys):
        for j, ba in enumerate(bobs[k].barcodes):
            all_bc[i, j] = np.array(ba.tolist(), dtype=np.float32)
    all_bc_flat = all_bc.reshape(N * M, L)  # (N*M, L)

    # cdist 'hamming' returns *fraction* of positions that differ; scale by L.
    NM = N * M
    if NM <= chunk:
        ham_flat = scipy_cdist(all_bc_flat, all_bc_flat, metric="hamming") * L
    else:
        ham_flat = np.empty((NM, NM), dtype=np.float32)
        for start in range(0, NM, chunk):
            end = min(start + chunk, NM)
            ham_flat[start:end] = (
                scipy_cdist(all_bc_flat[start:end], all_bc_flat, metric="hamming") * L
            )

    # Reshape (N*M, N*M) → (N, M, N, M) and reduce.
    ham_4d = ham_flat.reshape(N, M, N, M)
    # min over gallery barcodes (axis 3) → (N, M, N)
    min_per_qbc = ham_4d.min(axis=3)
    # median over query barcodes (axis 1) → (N, N)
    bob_dist_mat = np.median(min_per_qbc, axis=1)

    total_dist_mat = bob_dist_mat + semantic_weight * sem_dist_mat  # (N, N)

    distances: Dict[str, List] = defaultdict(list)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i != j:
                distances[ki].append((float(total_dist_mat[i, j]), kj))

    return distances
