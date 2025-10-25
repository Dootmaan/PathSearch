import torch, argparse, os, numpy as np
from tqdm import tqdm
from rich import print as rprint
from collections import defaultdict
from scipy import stats
import bitarray
from bitarray import util as butil

from PathSearch.dataset.Camelyon16Dataset import Camelyon16Dataset
from PathSearch.model.PathSearch import PathSearch

parser = argparse.ArgumentParser(description='CAMELYON16 Test (PathSearch + BoB)')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='./data/CAMELYON16')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--sample_num', type=int, default=-1)
params = parser.parse_args()
params.model_type = 'pathsearch'

device = torch.device(params.device)

testset = Camelyon16Dataset(path=params.data_dir, mode='test')

model = PathSearch(768).to(device)
if 'best' in params.model_path and params.model_path:
    model.load_state_dict(torch.load(params.model_path, map_location=device))
elif 'checkpoint' in params.model_path and params.model_path:
    model.load_state_dict(torch.load(params.model_path, map_location=device)['model'])
model.eval()

from open_clip import get_tokenizer
_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

from torch.utils.data import DataLoader
loader = DataLoader(testset, batch_size=params.batch_size, shuffle=False, num_workers=4, pin_memory=True)

all_img_feats, all_txt_feats, all_labels = [], [], []
all_mosaics, all_case_ids = [], []

with torch.no_grad():
    for case_id, img, label, coords, patch in tqdm(loader, desc='Encoding'):
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]
        img = img.to(device)
        # image/text features
        txt = _tokenizer(["dummy"], context_length=256).to(device)
        txt_feat = model.encode_text(txt)
        img_feat, mosaic = model.encode_image(img)
        all_case_ids.append(case_id)
        all_img_feats.append(img_feat.cpu())
        all_txt_feats.append(txt_feat.cpu())
        all_labels.append(torch.tensor([label]))
        mosaic = torch.cat((mosaic.cpu().squeeze(0), img_feat.cpu()), dim=0)
        all_mosaics.append(mosaic)

# --- BoB retrieval (enabled) ---
class BoB:
    def __init__(self, barcodes, semantic_embedding, name, site):
        import bitarray
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
