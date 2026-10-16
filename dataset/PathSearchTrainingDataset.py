import os
import glob
import pickle
from collections import defaultdict
from typing import Optional
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

class PathSearchDataset(torch.utils.data.Dataset):
    """PathSearch/CONCH-style dataset using ./data root and local splits/labels/texts.

    Root (default ./data/TCGA):
      ./splits/<...>/splits_0.csv         (auto-discovered, or pass split_files=)
      ./labels/BRCA_subtyping.csv         (or pass brca_csv=)
      ./texts/TCGA_all_clean_qianwen2.csv (or pass text_csv=)
      ./pt_files/conch_v1_5/*.pt
      ./h5_files/conch_v1_5/*.h5
      ./pt_files/<CANCER>/conch/*.pt      (for label discovery, or pass cancer_pt_dirs=)

    All path arguments that remain ``None`` fall back to ``data_root``-relative
    defaults, so the dataset works out-of-the-box when data is arranged as
    documented above without requiring the user to edit any source file.
    """
    def __init__(
        self,
        data_root: str = './data/TCGA',
        mode: str = 'test',
        sample_num: int = 512,
        cache_dir: Optional[str] = './kmeans_cache',
        rebuild_cache: bool = False,
        # Configurable path overrides — all default to data_root-relative locations.
        split_files: Optional[list] = None,
        cancer_pt_dirs: Optional[dict] = None,
        brca_csv: Optional[str] = None,
        text_csv: Optional[str] = None,
        text_col: int = 5,
    ):
        if cache_dir is None:
            cache_dir = './kmeans_cache'
        self.label, self.coords, self.patch_size_lv0 = [], [], []
        self.coords_h5_path, self.feature_pt_path, self.txts = [], [], []
        self.case_id = []
        self.mode = mode
        self.sample_num = int(sample_num)
        self.cache_dir = Path(cache_dir)
        self.kmeans_cache = {}

        # Initialize tokenizer (like PathVLM)
        from open_clip import get_tokenizer
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_data_paths(
            data_root,
            split_files=split_files,
            cancer_pt_dirs=cancer_pt_dirs,
            brca_csv=brca_csv,
            text_csv=text_csv,
            text_col=text_col,
        )

        cache_file = self.cache_dir / f"{mode}_cache.pkl"
        if not rebuild_cache and cache_file.exists():
            try:
                self._load_cache(cache_file)
            except ModuleNotFoundError as e:
                logging.warning("Cache loading failed due to: %s. Rebuilding cache...", e)
                self._preprocess_samples()
                self._save_cache(cache_file)
        else:
            self._preprocess_samples()
            self._save_cache(cache_file)

    def _load_data_paths(
        self,
        data_root: str,
        split_files: Optional[list] = None,
        cancer_pt_dirs: Optional[dict] = None,
        brca_csv: Optional[str] = None,
        text_csv: Optional[str] = None,
        text_col: int = 5,
    ):
        root = Path(data_root)

        # ---- Split files -------------------------------------------------------
        # If not provided, auto-discover under data_root/splits/*/splits_0.csv.
        if split_files is None:
            split_files = sorted(glob.glob(str(root / 'splits' / '*' / 'splits_0.csv')))
            if not split_files:
                logging.warning(
                    "No split files found under %s/splits/. All .pt files will be "
                    "treated as training data.", data_root
                )

        train_cases, val_cases, test_cases = set(), set(), set()
        for split_file in split_files:
            if os.path.exists(split_file):
                df = pd.read_csv(split_file, index_col=0, header=0)
                # Column 0 = train, Column 1 = val, Column 2 = test
                train_cases.update(df.iloc[:, 0].dropna().tolist())
                val_cases.update(df.iloc[:, 1].dropna().tolist())
                test_cases.update(df.iloc[:, 2].dropna().tolist())
            else:
                logging.warning("Split file not found: %s", split_file)

        # Select target cases based on mode
        if self.mode == 'train':
            pt_root = root / 'pt_files' / 'conch_v1_5'
            all_pts = glob.glob(str(pt_root / '*.pt'))
            target_cases = set(os.path.basename(x)[:-3] for x in all_pts) - (val_cases | test_cases)
        elif self.mode == 'val':
            target_cases = val_cases
        else:
            target_cases = test_cases

        # ---- Cancer-specific pt directories for label discovery ----------------
        # Default: data_root/pt_files/<CANCER>/conch/*.pt
        _default_cancer_types = ['LUAD', 'LUSC', 'KIRP', 'KIRC', 'KICH']
        if cancer_pt_dirs is None:
            cancer_pt_dirs = {
                c: str(root / 'pt_files' / c / 'conch') for c in _default_cancer_types
            }

        def _collect_cases(path: str) -> list:
            if os.path.exists(path):
                return [
                    os.path.basename(x).split('.pt')[0][:12]
                    for x in glob.glob(os.path.join(path, '*.pt'))
                ]
            return []

        LUAD_cases = _collect_cases(cancer_pt_dirs.get('LUAD', ''))
        LUSC_cases = _collect_cases(cancer_pt_dirs.get('LUSC', ''))
        KICH_cases = _collect_cases(cancer_pt_dirs.get('KICH', ''))
        KIRC_cases = _collect_cases(cancer_pt_dirs.get('KIRC', ''))
        KIRP_cases = _collect_cases(cancer_pt_dirs.get('KIRP', ''))

        # ---- BRCA subtyping CSV ------------------------------------------------
        if brca_csv is None:
            brca_csv = str(root / 'labels' / 'BRCA_subtyping.csv')
        ILC_cases, IDC_cases = [], []
        if os.path.exists(brca_csv):
            brca = pd.read_csv(brca_csv, index_col=2)
            for case_name in brca.index:
                lv = brca.loc[case_name][2]
                if lv == 'IDC':
                    IDC_cases.append(case_name[:12])
                elif lv == 'ILC':
                    ILC_cases.append(case_name[:12])
        else:
            logging.warning("BRCA subtyping CSV not found at %s; IDC/ILC labels will be empty.", brca_csv)

        # ---- Text (pathology report) CSV ---------------------------------------
        if text_csv is None:
            text_csv = str(root / 'texts' / 'TCGA_all_clean_qianwen2.csv')
        if not os.path.exists(text_csv):
            raise FileNotFoundError(
                f"Text CSV not found: {text_csv}\n"
                "Place the report CSV at data_root/texts/TCGA_all_clean_qianwen2.csv "
                "or pass text_csv= explicitly."
            )
        text_all = pd.read_csv(text_csv)
        all_case_ids = text_all.iloc[:, 0]
        logging.info("Text CSV columns: %s", list(text_all.columns))
        logging.info("Text CSV shape: %s", text_all.shape)
        text_col_idx = text_col

        # pt mapping
        pt_root = root / 'pt_files' / 'conch_v1_5'
        all_pt_files = glob.glob(str(pt_root / '*.pt'))
        case_to_pt = defaultdict(list)
        for pt_file in all_pt_files:
            filename = os.path.basename(pt_file)
            case_id = '-'.join(filename.split('-')[:3])
            case_to_pt[case_id].append(pt_file)

        valid = 0
        skipped_no_text = 0
        skipped_no_pt = 0
        skipped_not_in_target = 0
        
        for i in tqdm(range(len(all_case_ids)), desc=f"Processing {self.mode} data"):
            ccid = all_case_ids[i]
            text_value = text_all.iat[i, text_col_idx]
            
            # Check if text is valid (not nan and is string)
            if pd.isna(text_value) or not isinstance(text_value, str):
                skipped_no_text += 1
                if skipped_no_text <= 3:
                    logging.warning("[Skipped] text invalid: %s | %s", text_value, ccid)
                continue
                
            pt_files = sorted(case_to_pt.get(ccid, []))
            if not pt_files:
                skipped_no_pt += 1
                continue
            for pt_file in pt_files:
                pt_name = os.path.basename(pt_file)[:-3]
                if pt_name not in target_cases:
                    skipped_not_in_target += 1
                    continue
                if ccid in LUSC_cases:
                    label = 0
                elif ccid in LUAD_cases:
                    label = 1
                elif ccid in KICH_cases:
                    label = 2
                elif ccid in KIRC_cases:
                    label = 3
                elif ccid in KIRP_cases:
                    label = 4
                elif ccid in IDC_cases:
                    label = 5
                elif ccid in ILC_cases:
                    label = 6
                else:
                    if self.mode == 'train':
                        label = -1
                    else:
                        continue
                self.label.append(label)
                self.feature_pt_path.append(pt_file)
                self.case_id.append(pt_name)
                valid += 1
                # Tokenize text (like PathVLM)
                tokenized = self.tokenizer(text_value, context_length=256)
                self.txts.append(tokenized)
                coord_path = root / 'h5_files' / 'conch_v1_5' / f'{pt_name}.h5'
                self.coords_h5_path.append(str(coord_path))
                self.patch_size_lv0.append(np.int64(448))
        
        logging.info('Loaded %d valid samples', valid)
        logging.info('Skipped: no_text=%d, no_pt=%d, not_in_target=%d', skipped_no_text, skipped_no_pt, skipped_not_in_target)
        print(f'Loaded {valid} valid samples')

    def _preprocess_samples(self):
        logging.info("Starting dynamic proportional sampling...")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._process_single_sample, ch5, fpt, self.sample_num)
                for ch5, fpt in zip(self.coords_h5_path, self.feature_pt_path)
            ]
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
                
                # Fix: handle edge case when cluster is too small
                if n_rand > 0:
                    if len(sorted_idx) - n_close >= n_rand:
                        pool = sorted_idx[n_close:]
                        rand_samples = pool[torch.randperm(len(pool))[:n_rand]]
                    elif len(sorted_idx) > n_close:
                        # Sample with replacement from remaining samples
                        rand_samples = sorted_idx[torch.randint(n_close, len(sorted_idx), (n_rand,))]
                    else:
                        # If cluster only has n_close samples, duplicate some
                        rand_samples = sorted_idx[torch.randint(0, len(sorted_idx), (n_rand,))]
                    sampled.extend(torch.cat([close_samples, rand_samples]))
                else:
                    sampled.extend(close_samples)
            final_idx = torch.stack(sampled)[:sample_num]
            if len(final_idx) < sample_num:
                pad = sample_num - len(final_idx)
                final_idx = torch.cat([final_idx, torch.randint(0, total, (pad,))])
            return (feature_pt_path, data[final_idx].clone().cpu(), coords[final_idx])
        except Exception:
            import traceback
            logging.exception("Error processing %s:\n%s", feature_pt_path, traceback.format_exc())
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
            self.txts = cache['txts']  # Load texts from cache
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
            logging.warning("Error loading sample %s: %s", idx, e)
            return self.__getitem__((idx + 1) % len(self))

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader.
        
        Args:
            batch: List of tuples (case_id, data, text, label, coord, patch)
        
        Returns:
            Tuple of batched tensors
        """
        case_ids, datas, texts, labels, coords, patches = zip(*batch)
        
        # Stack data tensors (assuming they all have the same shape after kmeans)
        datas = torch.stack(datas, dim=0)
        
        # Stack tokenized texts (texts are already tensors from tokenizer)
        texts = torch.stack(texts, dim=0)
        
        # Convert labels to tensor
        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, dim=0)
        else:
            labels = torch.tensor(labels)
        
        return case_ids, datas, texts, labels, coords, patches
