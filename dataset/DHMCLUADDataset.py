import os
import torch
import random
import numpy as np
import pandas
import logging
import h5py
from typing import Optional
from fast_pytorch_kmeans import KMeans

class DHMCLUADDataset(torch.utils.data.Dataset):
    def __init__(self, path: str = './data/DHMC_LUAD', mode: str = 'train', method: str = 'conch_v1_5', sample_num: Optional[int] = None):
        super().__init__()
        self.mode = mode
        self.data = []
        self.label = []
        self.case_id = []
        self.coords = []
        self.patch_size_lv0 = []

        self.sample_num = sample_num
        self.kmeans = KMeans(n_clusters=sample_num, mode='euclidean', verbose=0) if sample_num is not None else None
        self.method = method

        labels_path = os.path.join(path, 'DHMC-LUAD.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"DHMC-LUAD labels CSV not found: {labels_path}")
        labels = pandas.read_csv(labels_path, index_col=1)
        test_cases = labels.iloc[:, 1].dropna().to_numpy().tolist()

        if self.mode == 'test':
            label_map = {'acinar': 0, 'solid': 1, 'lepidic': 2, 'papillary': 3, 'micropapillary': 4}
            for case in test_cases:
                feat_pt = os.path.join(path, 'pt_files', self.method, f"{case}.pt")
                logging.info('processing: %s', feat_pt)
                try:
                    npy = torch.load(feat_pt)
                    if len(npy) == 0:
                        continue
                except (OSError, RuntimeError) as e:
                    logging.warning('failed to load %s: %s', feat_pt, e)
                    continue

                try:
                    label = labels.loc[case][2]
                except KeyError as e:
                    logging.warning('label lookup failed for %s: %s', case, e)
                    continue
                if label not in label_map:
                    logging.warning('unknown class for %s: %s', case, label)
                    continue

                self.label.append(label_map[label])
                self.data.append(npy)
                self.case_id.append(case)

                coord_path = os.path.join(path, 'h5_files', self.method, f"{case}.h5")
                try:
                    with h5py.File(coord_path, 'r') as f:
                        coords = torch.from_numpy(f['coords'][:])
                    self.coords.append(coords)
                except (OSError, KeyError, ValueError) as e:
                    logging.warning('missing coords for %s: %s', coord_path, e)
                    self.coords.append(torch.empty((0, 2), dtype=torch.int64))

                self.patch_size_lv0.append(512)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.sample_num is None:
            return self.case_id[index], self.data[index], self.label[index], self.coords[index], self.patch_size_lv0[index]
        else:
            data1 = self.data[index]
            if len(data1) > self.sample_num:
                _ = self.kmeans.fit_predict(data1)
                data1 = self.kmeans.centroids
            else:
                data1 = torch.tensor(np.array(random.choices(data1.tolist(), k=self.sample_num)))
            return self.case_id[index], data1, self.label[index], self.coords[index], self.patch_size_lv0[index]
