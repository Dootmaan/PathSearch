import os
import glob
import pickle
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich import print as rprint

class TCGARetrievalUniversalDataset(torch.utils.data.Dataset):
    """Refactored to use ./data/* layout with no hard-coded absolute paths.

    Expected tree under `data_root` (default ./data/TCGA):
      ./splits/ <dataset> /splits_0.csv            # LUAD_LUSC_100, RCC_100, TCGA_BRCA_subtyping_100
      ./labels/BRCA_subtyping.csv
      ./texts/TCGA_all_clean_qianwen2.csv
      ./pt_files/conch_v1_5/*.pt
      ./h5_files/conch_v1_5/*.h5
    """
    def __init__(self, data_root: str = './data/TCGA', mode: str = 'test', sample_num: int = 512, cache_dir: str = './kmeans_cache', rebuild_cache: bool = False):
        self.label, self.coords, self.patch_size_lv0 = [], [], []
        self.coords_h5_path, self.feature_pt_path, self.txts = [], [], []
        self.case_id = []
        self.mode = mode
        self.sample_num = int(sample_num)
        self.cache_dir = Path(cache_dir)
        self.kmeans_cache = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_data_paths(data_root)

        cache_file = self.cache_dir / f"{mode}_cache.pkl"
        if not rebuild_cache and cache_file.exists():
            try:
                self._load_cache(cache_file)
            except ModuleNotFoundError as e:
                print(f"[Warning] Cache loading failed: {e}. Rebuilding...")
                self._preprocess_samples()
                self._save_cache(cache_file)
        else:
            self._preprocess_samples()
            self._save_cache(cache_file)

    def _load_data_paths(self, data_root: str):
        data_root = Path(data_root)
        # splits
        split_files = [
            data_root / 'splits' / 'LUAD_LUSC_100' / 'splits_0.csv',
            data_root / 'splits' / 'RCC_100' / 'splits_0.csv',
            data_root / 'splits' / 'TCGA_BRCA_subtyping_100' / 'splits_0.csv',
        ]
        train_cases, val_cases, test_cases = set(), set(), set()
        for sf in split_files:
            if not sf.exists():
                raise FileNotFoundError(f"Split file missing: {sf}")
            df = pd.read_csv(sf, index_col=0, header=0)
            train_cases.update(df.iloc[:, 0].dropna().tolist())
            val_cases.update(df.iloc[:, 1].dropna().tolist())
            test_cases.update(df.iloc[:, 2].dropna().tolist())
        if self.mode == 'train':
            target_cases = train_cases
            exclude_cases = val_cases | test_cases
        elif self.mode == 'val':
            target_cases = val_cases
            exclude_cases = train_cases | test_cases
        else:
            target_cases = test_cases
            exclude_cases = train_cases | val_cases

        # cancer-specific case id lists from pt directories (optional)
        def _cases_from_dir(root: Path):
            if not root.exists():
                return []
            return [os.path.basename(x).split('.pt')[0][:12] for x in glob.glob(str(root / '*.pt'))]

        LUAD_cases = _cases_from_dir(data_root / 'TCGA__LUAD' / 'pt_files' / 'conch_v1_5')
        LUSC_cases = _cases_from_dir(data_root / 'TCGA__LUSC' / 'pt_files' / 'conch_v1_5')
        KICH_cases = _cases_from_dir(data_root / 'TCGA__KICH' / 'pt_files' / 'conch_v1_5')
        KIRC_cases = _cases_from_dir(data_root / 'TCGA__KIRC' / 'pt_files' / 'conch_v1_5')
        KIRP_cases = _cases_from_dir(data_root / 'TCGA__KIRP' / 'pt_files' / 'conch_v1_5')

        # BRCA subtype labels
        brca_csv = data_root / 'labels' / 'BRCA_subtyping.csv'
        ILC_cases, IDC_cases = [], []
        if brca_csv.exists():
            brca = pd.read_csv(brca_csv, index_col=2)
            for case_name in brca.index:
                lv = brca.loc[case_name][2]
                (IDC_cases if lv == 'IDC' else ILC_cases if lv == 'ILC' else [])
            for case_name in brca.index:
                lv = brca.loc[case_name][2]
                if lv == 'IDC':
                    IDC_cases.append(case_name[:12])
                elif lv == 'ILC':
                    ILC_cases.append(case_name[:12])

        # text CSV
        txt_csv = data_root / 'texts' / 'TCGA_all_clean_qianwen2.csv'
        if not txt_csv.exists():
            raise FileNotFoundError(f"Text CSV not found: {txt_csv}")
        text_all = pd.read_csv(txt_csv)
        all_case_ids = text_all.iloc[:, 0]

        # map case_id -> pt files
        pt_root = data_root / 'pt_files' / 'conch_v1_5'
        all_pt_files = glob.glob(str(pt_root / '*.pt'))
        case_to_pt = defaultdict(list)
        for pt_file in all_pt_files:
            filename = os.path.basename(pt_file)
            case_id = '-'.join(filename.split('-')[:3])
            case_to_pt[case_id].append(pt_file)

        valid_count = 0
        for i in tqdm(range(len(all_case_ids)), desc=f"Processing {self.mode} data"):
            current_case_id = all_case_ids[i]
            if not isinstance(text_all.iat[i, 5], str):
                print(f"[Error] text not a string: {text_all.iat[i,5]} from case: {current_case_id}")
                continue
            pt_files = sorted(case_to_pt.get(current_case_id, []))
            if not pt_files:
                continue
            for pt_file in pt_files:
                pt_name = os.path.basename(pt_file)[:-3]
                if pt_name not in target_cases:
                    continue
                if current_case_id in LUSC_cases:
                    label = 0
                elif current_case_id in LUAD_cases:
                    label = 1
                elif current_case_id in KICH_cases:
                    label = 2
                elif current_case_id in KIRC_cases:
                    label = 3
                elif current_case_id in KIRP_cases:
                    label = 4
                elif current_case_id in IDC_cases:
                    label = 5
                elif current_case_id in ILC_cases:
                    label = 6
                else:
                    if self.mode == 'train':
                        label = -1
                    else:
                        continue
                self.label.append(label)
                self.feature_pt_path.append(pt_file)
                self.case_id.append(pt_name)
                valid_count += 1
                self.txts.append(text_all.iat[i, 5])
                coord_path = data_root / 'h5_files' / 'conch_v1_5' / f'{pt_name}.h5'
                self.coords_h5_path.append(str(coord_path))
                self.patch_size_lv0.append(np.int64(448))

        rprint('The distribution of the dataset is:')
        rprint('LUAD:', len([x for x in self.label if x == 1]))
        rprint('LUSC:', len([x for x in self.label if x == 0]))
        rprint('KICH:', len([x for x in self.label if x == 2]))
        rprint('KIRC:', len([x for x in self.label if x == 3]))
        rprint('KIRP:', len([x for x in self.label if x == 4]))
        rprint('IDC:', len([x for x in self.label if x == 5]))
        rprint('ILC:', len([x for x in self.label if x == 6]))
        rprint('Total:', len(self.label))
        rprint('-----------------------------------')
        rprint(f"Loaded {valid_count} valid samples | Mode: {self.mode} | Target: {len(target_cases)} | Excluded: {len(exclude_cases)}")

    def _preprocess_samples(self):
        print("\nStarting dynamic proportional sampling...")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(self._process_single_sample, ch5, fpt, self.sample_num) for ch5, fpt in zip(self.coords_h5_path, self.feature_pt_path)]
            for fut in tqdm(futures, desc="Processing samples"):
                result = fut.result()
                if result:
                    fpt, processed, coords = result
                    self.kmeans_cache[fpt] = processed
                    self.coords.append(coords)

    @staticmethod
    def _process_single_sample(coords_h5_path, feature_pt_path, sample_num):
        try:
            import h5py
            from fast_pytorch_kmeans import KMeans
            data = torch.load(feature_pt_path, weights_only=True)
            total = len(data)
            with h5py.File(coords_h5_path, 'r') as f:
                coords = torch.from_numpy(f['coords'][:])
            if total <= sample_num:
                idx = np.random.choice(total, sample_num, replace=True)
                return (feature_pt_path, data[idx].clone().cpu(), coords[idx])
            if sample_num == -1:
                return (feature_pt_path, data.clone().cpu(), coords)
            k = 16
            kmeans = KMeans(n_clusters=k, mode='euclidean')
            labels = kmeans.fit_predict(data)
            centroids = kmeans.centroids
            counts = torch.bincount(labels, minlength=k)
            base = torch.ones(k, dtype=torch.int32)
            remaining = sample_num - k
            prop = counts.float() / counts.sum()
            add = (prop * remaining).floor().int()
            rem = (prop * remaining) - add
            _, top_idx = torch.topk(rem, remaining - add.sum())
            add[top_idx] += 1
            quota = base + add
            cur_total = quota.sum()
            if cur_total > sample_num:
                excess = cur_total - sample_num
                _, reduce_idx = torch.topk(quota, excess)
                quota[reduce_idx] -= 1
            elif cur_total < sample_num:
                deficit = sample_num - cur_total
                quota[torch.argmax(quota)] += deficit
            sampled = []
            dists = torch.norm(data - centroids[labels], dim=1, p=2)
            for cid in range(k):
                mask = labels == cid
                idxs = torch.where(mask)[0]
                if len(idxs) == 0:
                    continue
                q = quota[cid].item()
                sorted_idx = idxs[dists[mask].sort().indices]
                n_close = max(1, int(q * 0.5))
                n_rand = q - n_close
                close_samples = sorted_idx[:n_close]
                if len(sorted_idx) - n_close >= n_rand:
                    pool = sorted_idx[n_close:]
                    rand_samples = pool[torch.randperm(len(pool))[:n_rand]]
                else:
                    rand_samples = sorted_idx[torch.randint(n_close, len(sorted_idx), (n_rand,))]
                sampled.extend(torch.cat([close_samples, rand_samples]))
            final_idx = torch.stack(sampled)[:sample_num]
            if len(final_idx) < sample_num:
                pad = sample_num - len(final_idx)
                final_idx = torch.cat([final_idx, torch.randint(0, total, (pad,))])
            return (feature_pt_path, data[final_idx].clone().cpu(), coords[final_idx])
        except Exception:
            import traceback
            print(f"Error processing {feature_pt_path}:\n{traceback.format_exc()}")
            return None

    def _save_cache(self, cache_file: Path):
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'kmeans_cache': self.kmeans_cache,
                'feature_pt_path': self.feature_pt_path,
                'txts': self.txts,
                'case_id': self.case_id,
                'label': self.label,
                'coords': self.coords,
                'patch_size_lv0': self.patch_size_lv0,
                'coords_h5_path': self.coords_h5_path,
            }, f)

    def _load_cache(self, cache_file: Path):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
            self.kmeans_cache = cache['kmeans_cache']
            self.feature_pt_path = cache['feature_pt_path']
            self.txts = cache['txts']
            self.case_id = cache['case_id']
            self.label = cache['label']
            self.coords = cache['coords']
            self.patch_size_lv0 = cache['patch_size_lv0']
            self.coords_h5_path = cache['coords_h5_path']

    def __len__(self):
        return len(self.txts)

    def __getitem__(self, idx):
        try:
            fpt = self.feature_pt_path[idx]
            text = self.txts[idx]
            case_id = self.case_id[idx]
            data = self.kmeans_cache[fpt]
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data)
            data = data.float()
            label = self.label[idx]
            coord = self.coords[idx]
            patch = self.patch_size_lv0[idx]
            return case_id, data, text, label, coord, patch
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
