import os
import glob
import logging
import pandas
import torch
import h5py

class Camelyon17Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str = './data', mode: str = 'train'):
        """Embedded feature dataset for CAMELYON16/17 under ./data.

        Layout:
          ./CAMELYON16/pt_files/conch_v1_5/*.pt
          ./CAMELYON17/pt_files/conch_v1_5/*.pt
          ./CAMELYON16/h5_files/conch_v1_5/*.h5
          ./CAMELYON17/h5_files/conch_v1_5/*.h5
          ./CAMELYON/camelyon.csv
        """
        super().__init__()
        root = os.path.abspath(root)
        self.mode = mode
        self.data, self.label, self.coords, self.patch_size_lv0, self.case_ids = [], [], [], [], []

        f16 = sorted(glob.glob(os.path.join(root, 'CAMELYON16', 'pt_files', 'conch_v1_5', '*.pt')))
        f17 = sorted(glob.glob(os.path.join(root, 'CAMELYON17', 'pt_files', 'conch_v1_5', '*.pt')))
        filenames = f17 + f16

        labels_path = os.path.join(root, 'CAMELYON', 'camelyon.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Camelyon labels CSV not found at {labels_path}")
        labels = pandas.read_csv(labels_path, index_col=2, header=0)

        for fname in filenames:
            npy = torch.load(fname)
            self.data.append(npy)
            pt = os.path.basename(fname).replace('.pt', '')

            # coord path by source
            if os.path.exists(os.path.join(root, 'CAMELYON16', 'pt_files', 'conch_v1_5', f'{pt}.pt')):
                coord_path = os.path.join(root, 'CAMELYON16', 'h5_files', 'conch_v1_5', f'{pt}.h5')
            else:
                coord_path = os.path.join(root, 'CAMELYON17', 'h5_files', 'conch_v1_5', f'{pt}.h5')

            try:
                with h5py.File(coord_path, 'r') as f:
                    coords = torch.from_numpy(f['coords'][:])
            except (OSError, KeyError, ValueError) as e:
                logging.warning("Failed to read coords from %s: %s", coord_path, e)
                coords = torch.empty((0, 2), dtype=torch.int64)
            self.coords.append(coords)

            self.patch_size_lv0.append(512)
            self.case_ids.append(pt)
            label = labels.loc[pt][2]
            self.label.append(1 if label == 'tumor' else 0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.case_ids[index], self.data[index], self.label[index]
        return self.case_ids[index], self.data[index], self.label[index], self.coords[index], self.patch_size_lv0[index]
