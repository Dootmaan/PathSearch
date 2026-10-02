import torch, argparse, os, numpy as np
from tqdm import tqdm
from rich import print as rprint
from collections import defaultdict
from scipy import stats
from bitarray import util as butil
import bitarray

from PathSearch.dataset.AnonymousDatasetTemplate import AnonymousDataset
from PathSearch.model.PathSearch import PathSearch

parser = argparse.ArgumentParser(description='Anonymous Dataset Test (PathSearch + BoB)')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='./data/Anonymous')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--sample_num', type=int, default=-1)
params = parser.parse_args()
params.model_type = 'pathsearch'

device = torch.device(params.device)

dataset = AnonymousDataset(path=params.data_dir, mode='test', sample_num=None if params.sample_num==-1 else params.sample_num)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = PathSearch(768).to(device)
if 'best' in params.model_path and params.model_path:
    model.load_state_dict(torch.load(params.model_path, map_location=device))
elif 'checkpoint' in params.model_path and params.model_path:
    model.load_state_dict(torch.load(params.model_path, map_location=device)['model'])
model.eval()

from open_clip import get_tokenizer
_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

all_img_feats, all_txt_feats, all_labels = [], [], []
all_mosaics, all_case_ids = [], []

with torch.no_grad():
    for case_id, img, label, coords, patch in tqdm(loader, desc='Encoding'):
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]
        img = img.to(device)
        txt = _tokenizer(["dummy"], context_length=256).to(device)
        txt_feat = model.encode_text(txt)
        img_feat, mosaic = model.encode_image(img)
        all_case_ids.append(case_id)
        all_img_feats.append(img_feat.cpu())
        all_txt_feats.append(txt_feat.cpu())
        all_labels.append(torch.tensor([label]))
        mosaic = torch.cat((mosaic.cpu().squeeze(0), img_feat.cpu()), dim=0)
        all_mosaics.append(mosaic)

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

rprint("\n=== Img2Img Top-k Results ===")
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
        if k>1 and stats.mode(top_k_labels, keepdims=False)[0] == query_label:
            mv_correct += 1
    rprint({f"@{k}": correct/max(1,len(distances)), **({f"MV@{k}": mv_correct/max(1,len(distances))} if k>1 else {})})


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
    tokenized = _tokenizer(text, context_length=256).to(device)
    txt_feature = model.encode_text(tokenized)
    return case_id, slide_feature, mosaic, txt_feature, label

# --- Evaluation function ---
def evaluate_model(testset):
    # Initialize metrics dict (adds img2img and txt2txt tasks)
    metrics = {
        'img2txt': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'txt2img': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'img2img': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'txt2txt': {'recall@1':0, 'recall@3':0, 'recall@5':0, 'vote_recall@3':0, 'vote_recall@5':0},
        'num_samples':0
    }

    # Store all features (image and text)
    all_img_feats, all_txt_feats, all_labels = [], [], []
    all_mosaics = []
    all_case_ids = []
    
    # Use DataLoader to speed up loading
    from torch.utils.data import DataLoader
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Feature collection phase
    with torch.no_grad():
        for datas in tqdm(test_loader, desc="Generating features"):
            if params.model_type == 'pathsearch':
                case_id, img_feats, mosaic, labels = generate_features(datas)
                txt_feats = img_feats
                all_case_ids.append(case_id)
                mosaic_cluster_npy = img_feats
                # if os.path.exists('/jhcnas4/hongyi/wsi-report-corrected-20250105/yottixel_mosaics/'+case_id.split(r'.')[0]+'/conch_v1.5_feats.npy'):
                #     # load corresponding mosaic features
                #     mosaic_cluster_npy = np.load('/jhcnas4/hongyi/wsi-report-corrected-20250105/yottixel_mosaics/'+case_id.split(r'.')[0]+'/conch_v1.5_feats.npy')
                #     mosaic_cluster_npy = torch.from_numpy(mosaic_cluster_npy)
                mosaic = torch.cat((mosaic.cpu().squeeze(0), mosaic_cluster_npy), dim=0)
                # mosaic = torch.cat((mosaic_cluster_npy, img_feats), dim=0)
                all_mosaics.append(mosaic)
                all_img_feats.append(img_feats.cpu())
                all_txt_feats.append(txt_feats.cpu())
                all_labels.append(labels.unsqueeze(0))
    
    if params.model_type == '':
        # Convert each mosaic feature into a BoB representation
        class BoB:
            def __init__(self, barcodes, semantic_embedding, name, site):
                self.barcodes = [bitarray.bitarray(b.tolist()) for b in barcodes]
                # self.barcodes = barcodes
                self.name = name
                self.site = site
                self.semantic_embedding = semantic_embedding
                
            def distance(self, bob):
                total_dist = []
                for feat in self.barcodes:
                    distances = [butil.count_xor(feat, b) for b in bob.barcodes]
                    # distances = [feat@b for b in bob.barcodes]
                    total_dist.append(np.min(distances))
                    # total_dist.append(np.max(distances))
                retval = np.median(total_dist)
                semantic_dist = torch.norm(self.semantic_embedding-bob.semantic_embedding, p=2, dim=1)
                return retval + 200*semantic_dist.item()

        BoBs = {}
        for i in range(len(all_mosaics)):
            mosaic_feat = all_mosaics[i]
            # mosaic_feat = torch.cat((mosaic_feat, all_img_feats[i]), dim=0)
            # barcodes = mosaic_feat
            barcodes = (np.diff(np.array(mosaic_feat), n=1, axis=1) < 0) * 1
            BoBs[all_case_ids[i]] = BoB(barcodes, all_img_feats[i], all_case_ids[i], all_labels[i])

        distances = defaultdict(list)
        for a in BoBs:
            for b in BoBs:
                if a == b: continue
                dist = BoBs[a].distance(BoBs[b])
                distances[a].append((dist, b))

    # Top-k evaluation
        topk_results = {}
        for k in [1, 3, 5]:
            correct = 0
            mv_correct = 0
            for query in distances:
                distances[query].sort(key=lambda x: x[0])
                top_k = distances[query][:k]
                top_k_labels = [BoBs[n].site.item() for _, n in top_k]
                query_label = BoBs[query].site.item()
                # Top-1
                if top_k_labels[0] == query_label:
                    correct += 1
                # majority voting
                if stats.mode(top_k_labels, keepdims=False)[0] == query_label:
                    mv_correct += 1
            topk_results[f"@{k}"] = correct / len(distances)
            if k > 1:
                topk_results[f"MV@{k}"] = mv_correct / len(distances)
                
        rprint("\n=== Img2Img Top-k Results ===")
        for k, v in topk_results.items():
            rprint(f"Top-k {k}: {v:.4f}")
        
    
    # Move to GPU
    img_feats = torch.cat(all_img_feats).to(device)
    txt_feats = torch.cat(all_txt_feats).to(device)
    all_labels = torch.cat(all_labels).to(device)  # (total_samples, )
    print(f"img_feats shape: {img_feats.shape}")
    print(f"txt_feats shape: {txt_feats.shape}")
    print(f"all_labels shape: {all_labels.shape}")

    # Chunked evaluation helper
    def update_metrics(sim_matrix, query_labels, gallery_labels, metric_name):
        """Unified evaluation helper for four task types"""
        # Exclude self retrieval cases (used for img2img and txt2txt)
        if metric_name in ['img2img', 'txt2txt']:
            mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        for k in [1,3,5]:
            _, topk_indices = torch.topk(sim_matrix, k, dim=-1)
            topk_labels = gallery_labels[topk_indices]  # (batch_size, k)
            
            # Check whether the top-k contains same-class samples
            correct = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
            
            # Use 'recall' as the metric name
            metrics[metric_name][f'recall@{k}'] += correct.sum().item()
            
            # Voting mechanism
            if k in [3,5]:
                vote_labels = torch.mode(topk_labels, dim=1).values
                vote_correct = (vote_labels == query_labels)
                metrics[metric_name][f'vote_recall@{k}'] += vote_correct.sum().item()

    # Chunked evaluation
    for idx in tqdm(range(0, len(all_labels), params.chunk_size), desc="Evaluating"):
        end_idx = min(idx+params.chunk_size, len(all_labels))
        
    # Current chunk
        img_chunk = img_feats[idx:end_idx]
        txt_chunk = txt_feats[idx:end_idx]
        label_chunk = all_labels[idx:end_idx]
        
        # === img2txt ===
        img2txt_sim = img_chunk @ txt_feats.T
        update_metrics(img2txt_sim, label_chunk, all_labels, 'img2txt')
        
        # === txt2img ===
        txt2img_sim = txt_chunk @ img_feats.T
        update_metrics(txt2img_sim, label_chunk, all_labels, 'txt2img')
        
        # === img2img ===
        img2img_sim = img_chunk @ img_feats.T
        update_metrics(img2img_sim, label_chunk, all_labels, 'img2img')
        
        # === txt2txt ===
        txt2txt_sim = txt_chunk @ txt_feats.T
        update_metrics(txt2txt_sim, label_chunk, all_labels, 'txt2txt')
        
        metrics['num_samples'] += (end_idx - idx)
    
    # Normalize metrics
    for metric in metrics:
        if metric != 'num_samples':
            for key in metrics[metric]:
                metrics[metric][key] /= metrics['num_samples']
    
    # Print final results (includes two additional tasks)
    print("\n=== Final Metrics ===")
    for task in ['img2txt', 'txt2img', 'img2img', 'txt2txt']:
        print(f"\n** {task.upper()} **")
        for k, v in metrics[task].items():
            print(f"{k}: {v:.4f}")
    
    # Save results (column order adjusted)
    df = pd.DataFrame([
        ['img2txt'] + list(metrics['img2txt'].values()),
        ['txt2img'] + list(metrics['txt2img'].values()),
        ['img2img'] + list(metrics['img2img'].values()),
        ['txt2txt'] + list(metrics['txt2txt'].values())
    ], columns=['Task', 'recall@1', 'recall@3', 'recall@5', 'vote_recall@3', 'vote_recall@5'])
    
    # Save results to the specified path
    if params.sample_num == -1:
        subroot = os.path.join('/home/hongyi/Workspace-Python/PathSearch/temp_result/all', params.model_path.split('/')[-3], params.model_path.split('/')[-2])
    # subroot = os.path.join('/home/hongyi/Workspace-Python/PathSearch/temp_result/all',params.model_path.split('/')[-2])
    else:
        subroot = os.path.join(f'/home/hongyi/Workspace-Python/PathSearch/temp_result/kmeans{params.sample_num}', params.model_path.split('/')[-2])
    if not os.path.exists(subroot):
        os.makedirs(subroot)
    result_filename = f"TCGA_results_{params.model_type}_{params.feature_method}_{params.model_path.split('/')[-1].split('.')[0]}.xlsx"
    df.to_excel(os.path.join(subroot, result_filename), index=False)
    return metrics

evaluate_model(dataset)