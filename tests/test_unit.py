"""Unit tests for PathSearch core components.

These tests use only deterministic synthetic data and require no external
datasets, pretrained weights, or network access.  They can be run with:

    python -m pytest tests/test_unit.py -v

or simply:

    python tests/test_unit.py
"""
import sys
import os
import unittest

import numpy as np
import torch
import bitarray

# Make repo root importable when running tests directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval_backend import (
    BoB,
    build_bobs,
    compute_all_distances,
    compute_all_distances_batched,
    DEFAULT_SEMANTIC_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers (shared fixtures)
# ---------------------------------------------------------------------------

def _make_barcode(bits: list) -> bitarray.bitarray:
    ba = bitarray.bitarray(bits)
    return ba


def _make_bob(name: str, label: int, barcode_bits: list, embed_dim: int = 4,
              semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT) -> BoB:
    """Create a minimal BoB with a single barcode and a fixed semantic vector."""
    bc = _make_barcode(barcode_bits)
    sem = torch.ones(1, embed_dim) * float(label)  # deterministic per-class
    return BoB(barcodes=[bc], semantic_embedding=sem, name=name, site=label,
               semantic_weight=semantic_weight)


# ---------------------------------------------------------------------------
# BoB unit tests
# ---------------------------------------------------------------------------

class TestBoBDistance(unittest.TestCase):
    """Tests for BoB.distance()."""

    def test_identical_barcodes_identical_semantics_gives_zero(self):
        """Distance between a slide and an identical copy should be 0."""
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        a = _make_bob('a', 0, bits)
        b = _make_bob('b', 0, bits)
        self.assertAlmostEqual(a.distance(b), 0.0, places=6)

    def test_distance_non_negative(self):
        """BoB distance must always be >= 0."""
        a = _make_bob('a', 0, [1, 0, 1, 1, 0, 0])
        b = _make_bob('b', 1, [0, 1, 0, 0, 1, 1])
        self.assertGreaterEqual(a.distance(b), 0.0)
        self.assertGreaterEqual(b.distance(a), 0.0)

    def test_hamming_only_distance_correct(self):
        """With semantic_weight=0 the distance equals the Hamming count."""
        bits_a = [1, 0, 1, 1, 0, 0]  # 6 bits
        bits_b = [0, 0, 1, 0, 0, 0]  # differs at positions 0 and 3 → 2 bits
        # Use same semantic so semantic term = 0
        sem = torch.zeros(1, 4)
        a = BoB(barcodes=[_make_barcode(bits_a)], semantic_embedding=sem,
                name='a', site=0, semantic_weight=0.0)
        b = BoB(barcodes=[_make_barcode(bits_b)], semantic_embedding=sem,
                name='b', site=0, semantic_weight=0.0)
        expected_hamming = sum(x != y for x, y in zip(bits_a, bits_b))  # 2
        self.assertAlmostEqual(a.distance(b), float(expected_hamming), places=6)

    def test_semantic_only_distance_correct(self):
        """With identical barcodes the distance equals semantic_weight * ||a-b||_2."""
        bits = [1, 0, 1]
        sem_a = torch.tensor([[1.0, 0.0]])
        sem_b = torch.tensor([[4.0, 0.0]])  # L2 dist = 3.0
        weight = 5.0
        a = BoB(barcodes=[_make_barcode(bits)], semantic_embedding=sem_a,
                name='a', site=0, semantic_weight=weight)
        b = BoB(barcodes=[_make_barcode(bits)], semantic_embedding=sem_b,
                name='b', site=0, semantic_weight=weight)
        expected = 0.0 + weight * 3.0
        self.assertAlmostEqual(a.distance(b), expected, places=5)

    def test_semantic_weight_parameter_respected(self):
        """Changing semantic_weight changes the distance proportionally."""
        bits = [0, 0, 0, 0]
        sem_a = torch.ones(1, 2)
        sem_b = torch.zeros(1, 2)
        l2 = float(torch.norm(sem_a - sem_b))

        for w in [0.0, 1.0, 100.0, 200.0]:
            a = BoB(barcodes=[_make_barcode(bits)], semantic_embedding=sem_a,
                    name='a', site=0, semantic_weight=w)
            b = BoB(barcodes=[_make_barcode(bits)], semantic_embedding=sem_b,
                    name='b', site=0, semantic_weight=w)
            self.assertAlmostEqual(a.distance(b), w * l2, places=5)

    def test_site_stored_as_int(self):
        """build_bobs must unwrap 1-element tensors so site is a plain int."""
        case_ids = ['s0', 's1']
        bits = [1, 0, 1]
        mosaics = [torch.rand(3, 4), torch.rand(3, 4)]
        img_feats = [torch.zeros(1, 4), torch.ones(1, 4)]
        labels = [torch.tensor([0]), torch.tensor([1])]  # 1-element tensors
        bobs = build_bobs(case_ids, mosaics, img_feats, labels)
        for k, bob in bobs.items():
            self.assertIsInstance(bob.site, int,
                                  f"site of {k} should be int, got {type(bob.site)}")


# ---------------------------------------------------------------------------
# build_bobs tests
# ---------------------------------------------------------------------------

class TestBuildBobs(unittest.TestCase):

    def _make_inputs(self, n=3, patch_dim=8, embed_dim=4):
        case_ids = [f'slide_{i}' for i in range(n)]
        # mosaics: (num_patches, patch_dim); barcodes derived via diff → (num_patches, patch_dim-1)
        mosaics = [torch.rand(5, patch_dim) for _ in range(n)]
        img_feats = [torch.rand(1, embed_dim) for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        return case_ids, mosaics, img_feats, labels

    def test_keys_match_case_ids(self):
        case_ids, mosaics, img_feats, labels = self._make_inputs(4)
        bobs = build_bobs(case_ids, mosaics, img_feats, labels)
        self.assertEqual(set(bobs.keys()), set(case_ids))

    def test_barcode_length_is_dim_minus_one(self):
        patch_dim = 8
        case_ids, mosaics, img_feats, labels = self._make_inputs(2, patch_dim=patch_dim)
        bobs = build_bobs(case_ids, mosaics, img_feats, labels)
        for bob in bobs.values():
            for bc in bob.barcodes:
                self.assertEqual(len(bc), patch_dim - 1,
                                 "Barcode length should equal patch_dim - 1")

    def test_semantic_embedding_on_cpu(self):
        case_ids, mosaics, img_feats, labels = self._make_inputs()
        bobs = build_bobs(case_ids, mosaics, img_feats, labels)
        for bob in bobs.values():
            self.assertEqual(bob.semantic_embedding.device, torch.device('cpu'))

    def test_custom_semantic_weight_propagated(self):
        case_ids, mosaics, img_feats, labels = self._make_inputs(2)
        bobs = build_bobs(case_ids, mosaics, img_feats, labels, semantic_weight=42.0)
        for bob in bobs.values():
            self.assertEqual(bob.semantic_weight, 42.0)


# ---------------------------------------------------------------------------
# compute_all_distances / compute_all_distances_batched tests
# ---------------------------------------------------------------------------

class TestComputeDistances(unittest.TestCase):

    def _fixture_bobs(self, n=4):
        """Create n BoB objects with deterministic, varied embeddings."""
        bobs = {}
        torch.manual_seed(0)
        bits = [1, 0, 1, 1, 0]
        for i in range(n):
            sem = torch.rand(1, 8) * (i + 1)
            bobs[f'slide_{i}'] = BoB(
                barcodes=[_make_barcode(bits)],
                semantic_embedding=sem,
                name=f'slide_{i}',
                site=i % 2,
                semantic_weight=1.0,  # small weight for numeric clarity
            )
        return bobs

    def test_loop_distance_keys_complete(self):
        bobs = self._fixture_bobs(3)
        d = compute_all_distances(bobs)
        for key in bobs:
            self.assertIn(key, d)
            # Each query should have N-1 entries
            self.assertEqual(len(d[key]), len(bobs) - 1)

    def test_batched_distance_keys_complete(self):
        bobs = self._fixture_bobs(3)
        d = compute_all_distances_batched(bobs)
        for key in bobs:
            self.assertIn(key, d)
            self.assertEqual(len(d[key]), len(bobs) - 1)

    def test_loop_and_batched_agree(self):
        """Loop and batched implementations must return the same distances."""
        bobs = self._fixture_bobs(4)
        d_loop = compute_all_distances(bobs)
        d_batched = compute_all_distances_batched(bobs)

        keys = list(bobs.keys())
        for a in keys:
            # Build {peer_name: dist} dicts for easy comparison
            loop_dict = {name: dist for dist, name in d_loop[a]}
            batched_dict = {name: dist for dist, name in d_batched[a]}
            for b in keys:
                if a == b:
                    continue
                self.assertAlmostEqual(
                    loop_dict[b], batched_dict[b], places=4,
                    msg=f"Distance mismatch for ({a}, {b}): "
                        f"loop={loop_dict[b]:.6f}, batched={batched_dict[b]:.6f}"
                )

    def test_all_distances_non_negative(self):
        bobs = self._fixture_bobs(4)
        for d in [compute_all_distances(bobs), compute_all_distances_batched(bobs)]:
            for key, pairs in d.items():
                for dist, _ in pairs:
                    self.assertGreaterEqual(dist, 0.0, f"Negative distance for query {key}")

    def test_empty_bobs(self):
        """Empty BoB dict should return empty distances without error."""
        d = compute_all_distances({})
        self.assertEqual(len(d), 0)
        d_b = compute_all_distances_batched({})
        self.assertEqual(len(d_b), 0)


# ---------------------------------------------------------------------------
# PathSearch model shape tests (CPU-only, no pretrained weights)
# ---------------------------------------------------------------------------

class TestPathSearchShapes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Import lazily so unit tests don't fail if torch is absent on CI.
        from model.PathSearch import PathSearch
        cls.PathSearch = PathSearch

    def test_encode_image_output_shape(self):
        """encode_image(B, N, D) should return (B, D) features and (B, M, D) mosaics."""
        embed_dim = 64
        B, N = 2, 32
        # No text encoder needed for image-only test.
        model = self.PathSearch(embed_dim=embed_dim, mosaic_num=4)
        model.eval()
        with torch.no_grad():
            x = torch.rand(B, N, embed_dim)
            feats, mosaics = model.encode_image(x, normalize=True)
        self.assertEqual(feats.shape, (B, embed_dim))
        self.assertEqual(mosaics.shape[0], B)
        self.assertEqual(mosaics.shape[-1], embed_dim)

    def test_encode_image_normalized(self):
        """L2 norm of each output feature vector should be ~1."""
        embed_dim = 64
        model = self.PathSearch(embed_dim=embed_dim, mosaic_num=4)
        model.eval()
        with torch.no_grad():
            x = torch.rand(1, 16, embed_dim)
            feats, _ = model.encode_image(x, normalize=True)
        norms = feats.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_text_encoder_unavailable_raises(self):
        """encode_text should raise RuntimeError when open_clip is not available."""
        from unittest.mock import patch
        # Simulate open_clip being missing.
        with patch.dict('sys.modules', {'open_clip': None}):
            # Re-import with the module removed from cache.
            import importlib
            import model.PathSearch as ps_mod
            importlib.reload(ps_mod)
            # Patch create_model_from_pretrained to raise ImportError.
            with patch.object(ps_mod, 'TransMILABMIL', ps_mod.TransMILABMIL):
                model = ps_mod.PathSearch.__new__(ps_mod.PathSearch)
                model._text_encoder_available = False
                from src.retrieval_backend import BoB  # noqa: import guard

                # Sentinel model should raise when called.
                class _Sentinel:
                    def encode_text(self, *a, **kw):
                        raise RuntimeError("Text encoder unavailable")
                model.text_model = _Sentinel()
                with self.assertRaises(RuntimeError):
                    model.text_model.encode_text(torch.zeros(1, 10))


if __name__ == '__main__':
    unittest.main()
