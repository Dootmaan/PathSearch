import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from transformers import AutoModel  # (kept import to minimize diffs, unused)
from rich import print as rprint
from collections import defaultdict
from scipy import stats
import bitarray
import numpy as np
from bitarray import util as butil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dataset.TCGATestingDataset import TCGARetrievalUniversalDataset

# --- Args (minimal change): remove --model_type, keep others; default data_dir -> ./data/TCGA
parser = argparse.ArgumentParser(description='Universal Model Testing for TCGA (PathSearch + BoB)')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--data_dir', type=str, default='./data/TCGA')
parser.add_argument('--model_path', type=str, default='./results/pathsearch_best.pt')
parser.add_argument('--feature_method', choices=['model', 'pooling'], default='model')
parser.add_argument('--chunk_size', type=int, default=600, help='Similarity matrix chunk size')
parser.add_argument('--sample_num', type=int, default=-1, help='Number of samples per slide (-1 uses all)')
params = parser.parse_args()

# --- Device
device = torch.device(params.device)

# --- Dataset (unchanged logic; paths now relative)
if params.sample_num == 512:
    testset = TCGARetrievalUniversalDataset(
        data_root=params.data_dir,
        mode='test',
        sample_num=params.sample_num,
        cache_dir='./cache_all/test_TCGA/kmeans_test_cache',
        rebuild_cache=False
    )
elif params.sample_num == -1:
    testset = TCGARetrievalUniversalDataset(
        data_root=params.data_dir,
        mode='test',
        sample_num=params.sample_num,
        cache_dir='./cache_all/test_TCGA/all_test_cache',
        rebuild_cache=False
    )

# --- PathSearch model branch only ---
from model.PathSearch import PathSearch
model = PathSearch(768).to(device)
# optional load
if os.path.isfile(params.model_path):
    state = torch.load(params.model_path, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
model.to(device)
from open_clip import get_tokenizer
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()

# --- Feature generation (PathSearch only)
@torch.no_grad()
def generate_features(datas):
    case_id, img, text, label, coords, patch_size = datas  # img: (N,768)
    case_id = case_id[0]
    img = img.to(device)
    coords = coords.to(device)
    label = torch.tensor(label)
    mosaic = None

    # cache dir -> relative
    from pathlib import Path
    if params.sample_num == -1:
        embed_dir = Path('./cache_all/test_TCGA/slide_embedding/all/default/default')
    else:
        embed_dir = Path(f'./cache_all/test_TCGA/slide_embedding/kmeans{params.sample_num}/default/default')
    embed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = embed_dir / f"{case_id}_embedding.pt"

    slide_feature, mosaic = model.encode_image(img)
    torch.save(slide_feature.detach().cpu(), cache_path)
    slide_feature = slide_feature.detach().cpu()
    # text feature via model (kept interface parity)
    tokenized = tokenizer(text, context_length=256).to(device)
    txt_feature = model.encode_text(tokenized)
    return case_id, slide_feature, mosaic, txt_feature, label

# --- Evaluation (kept original style + BoB enabled)
@torch.no_grad()
def evaluate_model(testset):
    metrics = {
        'img2txt': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'txt2img': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'img2img': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'txt2txt': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'num_samples':0
    }

    all_img_feats, all_txt_feats, all_labels = [], [], []
    all_mosaics, all_case_ids = [], []

    from torch.utils.data import DataLoader
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for datas in tqdm(test_loader, desc='Generating features'):
        case_id, img_feats, mosaic, txt_feats, labels = generate_features(datas)
        all_case_ids.append(case_id)
        mosaic_cluster_npy = img_feats
        # keep structure but use only local tensors
        mosaic = torch.cat((mosaic.cpu().squeeze(0), mosaic_cluster_npy, img_feats), dim=0)
        all_mosaics.append(mosaic)
        all_img_feats.append(img_feats.cpu())
        all_txt_feats.append(txt_feats.cpu())
        all_labels.append(labels.unsqueeze(0))

    # ---- BoB block (ENABLED) ----
    class BoB:
        def __init__(self, barcodes, semantic_embedding, name, site):
            self.barcodes = [bitarray.bitarray(b.tolist()) for b in barcodes]
            self.name = name
            self.site = site
            self.semantic_embedding = semantic_embedding
        def distance(self, bob):
            total_dist = []
            for feat in self.barcodes:
                distances = [butil.count_xor(feat, b) for b in bob.barcodes]
                total_dist.append(np.min(distances))
            retval = np.median(total_dist)
            semantic_dist = torch.norm(self.semantic_embedding-bob.semantic_embedding, p=2, dim=1)
            return retval + 200*semantic_dist.item()

    BoBs = {}
    for i in range(len(all_mosaics)):
        mosaic_feat = all_mosaics[i]
        barcodes = (np.diff(np.array(mosaic_feat), n=1, axis=1) < 0) * 1
        BoBs[all_case_ids[i]] = BoB(barcodes, all_img_feats[i], all_case_ids[i], all_labels[i])

    distances = defaultdict(list)
    for a in BoBs:
        for b in BoBs:
            if a == b: continue
            dist = BoBs[a].distance(BoBs[b])
            distances[a].append((dist, b))

    topk_results = {}
    for k in [1,3,5]:
        correct = 0
        mv_correct = 0
        for query in distances:
            distances[query].sort(key=lambda x: x[0])
            top_k = distances[query][:k]
            top_k_labels = [BoBs[n].site.item() for _, n in top_k]
            query_label = BoBs[query].site.item()
            if top_k_labels[0] == query_label:
                correct += 1
            if stats.mode(top_k_labels, keepdims=False)[0] == query_label:
                mv_correct += 1
        topk_results[f"@{k}"] = correct / len(distances)
        if k > 1:
            topk_results[f"MV@{k}"] = mv_correct / len(distances)
    rprint("\n=== Img2Img Top-k Results ===")
    for k, v in topk_results.items():
        rprint(f"Top-k {k}: {v:.4f}")

    # ---- Four-direction retrieval (format preserved) ----
    img_feats = torch.cat(all_img_feats).to(device)
    txt_feats = torch.cat(all_txt_feats).to(device)
    all_labels = torch.cat(all_labels).to(device)

    def update_metrics(sim_matrix, query_labels, gallery_labels, metric_name):
        if metric_name in ['img2img', 'txt2txt']:
            mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        for k in [1,3,5]:
            _, topk_indices = torch.topk(sim_matrix, k, dim=-1)
            topk_labels = gallery_labels[topk_indices]
            correct = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
            metrics[metric_name][f'recall@{k}'] += correct.sum().item()
            if k in [3,5]:
                vote_labels = torch.mode(topk_labels, dim=1).values
                vote_correct = (vote_labels == query_labels)
                metrics[metric_name][f'vote_recall@{k}'] += vote_correct.sum().item()

    # cosine sims
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    txt_feats = torch.nn.functional.normalize(txt_feats, dim=-1)
    sim_img_img = img_feats @ img_feats.t()
    sim_img_txt = img_feats @ txt_feats.t()
    sim_txt_img = txt_feats @ img_feats.t()
    sim_txt_txt = txt_feats @ txt_feats.t()

    update_metrics(sim_img_txt, all_labels, all_labels, 'img2txt')
    update_metrics(sim_txt_img, all_labels, all_labels, 'txt2img')
    update_metrics(sim_img_img, all_labels, all_labels, 'img2img')
    update_metrics(sim_txt_txt, all_labels, all_labels, 'txt2txt')

    n = img_feats.size(0)
    for task in metrics:
        if task == 'num_samples':
            continue
        for k in list(metrics[task].keys()):
            metrics[task][k] /= n
    rprint("\n=== Four-direction Retrieval (recall) ===")
    for task, vals in metrics.items():
        if task == 'num_samples':
            continue
        rprint(task, {k: f"{v:.4f}" for k, v in vals.items()})


evaluate_model(testset)
