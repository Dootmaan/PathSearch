# pip install git+https://github.com/Mahmoodlab/CONCH.git

from conch.open_clip_custom import create_model_from_pretrained
import torch
from peft import LoraConfig, get_peft_model
import argparse
import os
from pathlib import Path
from PathSearch import config


def get_model(device):
    """Return a CONCH ViT-B-16 visual encoder callable that maps images->embeddings.
    Uses optional custom weights if present; otherwise falls back to pretrained defaults.
    """
    # Resolve candidate paths in priority order
    candidates = []
    if config.MILIP_WEIGHTS_PATH:
        candidates.append(str(config.MILIP_WEIGHTS_PATH))
    candidates.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'ckpts', 'conch.pth'))
    candidates.append(os.path.join(config.REPO_ROOT, 'models', 'ckpts', 'conch.pth'))

    weights_path = None
    for p in candidates:
        if p and os.path.exists(p):
            weights_path = p
            break

    if weights_path is None:
        print('No custom conch weights found; falling back to pretrained defaults via create_model_from_pretrained')

    model, _ = create_model_from_pretrained('conch_ViT-B-16', weights_path, device=device)

    # Attempt to load custom checkpoint (non-fatal on mismatch)
    if weights_path is not None and os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f'custom weights loaded: {weights_path}')
        except Exception as e:
            print(f'Failed to load state_dict from {weights_path}: {e}')

    # Use only the visual encoder
    model = model.visual

    def func(image):
        with torch.no_grad():
            image_embs, _ = model(image)
            return image_embs

    return func


def get_conch_trans():
    """Return the default CONCH ViT-B-16 preprocessing transforms."""
    weights = None
    if config.MILIP_WEIGHTS_PATH and os.path.exists(str(config.MILIP_WEIGHTS_PATH)):
        weights = str(config.MILIP_WEIGHTS_PATH)
    return create_model_from_pretrained('conch_ViT-B-16', weights)[1]


def _load_conch_v1_5_from_hf():
    """Load CONCH v1.5 from Hugging Face without any dependency on upstream wrappers.

    We prefer the standalone CONCH v1.5 repository: "MahmoodLab/conchv1_5".
    Some builds expose a `.return_conch()` helper; others provide the model directly
    with a `.preprocess` attribute. We support both gracefully.
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")

    repo_id = 'MahmoodLab/conchv1_5'
    conch = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    model = None
    preprocess = None

    if hasattr(conch, 'return_conch') and callable(getattr(conch, 'return_conch')):
        model, preprocess = conch.return_conch()
    else:
        model = conch
        # Best-effort: many repos expose a preprocess callable/transform
        preprocess = getattr(conch, 'preprocess', None)

    return model, preprocess


def get_conch_v1_5_model(device):
    """Return the CONCH v1.5 vision model loaded from HF (MahmoodLab/conchv1_5)."""
    model, _ = _load_conch_v1_5_from_hf()

    # Ensure visual-only forward where applicable and typical normalization behaviour
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.eval()
    return model.to(device=device)


def get_conch_v1_5_trans():
    """Return the CONCH v1.5 preprocessing transforms from HF if available."""
    _, preprocess = _load_conch_v1_5_from_hf()
    if preprocess is None:
        raise RuntimeError('No preprocess transforms found for CONCH v1.5 (MahmoodLab/conchv1_5).')
    print(preprocess)
    return preprocess


def get_conch_v1_5_step3_epoch20_model(device):
    """Load CONCH v1.5 from HF and (optionally) inject LoRA + load a step3/epoch20 checkpoint.

    The checkpoint can be provided via env var `PATHSEARCH_STEP3_CHECKPOINT`.
    """
    model, _ = _load_conch_v1_5_from_hf()

    def add_lora_adapter(model: torch.nn.Module) -> torch.nn.Module:
        start_block = 18
        end_block = 23
        target_modules = [f"trunk.blocks.{i}.attn.qkv" for i in range(start_block, end_block + 1)]
        cfg = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[],
        )
        return get_peft_model(model, cfg)

    model = model.to(device)
    model = add_lora_adapter(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint_path = os.environ.get('PATHSEARCH_STEP3_CHECKPOINT', '')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print('No PATHSEARCH_STEP3_CHECKPOINT provided or file does not exist; returning uninitialized model.')
        model.eval()
        return model.to(device=device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', {}).get('conch_model', {})

    # Normalize possible "module." prefixes and nested keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k.replace('module.', '', 1)
        if new_k.startswith('base_model.model.'):
            new_k = new_k.replace('base_model.model.', '', 1)
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[LoRA] Missing keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[LoRA] Unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")

    print(f'custom checkpoint loaded from: {checkpoint_path}')
    model.eval()
    return model.to(device=device)


def get_conch_v1_5_step3_epoch20_trans():
    _, preprocess = _load_conch_v1_5_from_hf()
    if preprocess is None:
        raise RuntimeError('No preprocess transforms found for CONCH v1.5 step3_epoch20 (MahmoodLab/conchv1_5).')
    print(preprocess)
    return preprocess
