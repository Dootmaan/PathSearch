import os
import glob
import logging
import pandas
import torch
import h5py

class Camelyon16Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str = './data/CAMELYON16', mode: str = 'train'):
        """Embedded feature dataset for CAMELYON16 under ./data/CAMELYON16.

        Layout:
          ./pt_files/conch_v1_5/*.pt
          ./h5_files/conch_v1_5/*.h5
          ./camelyon16_official_split_mapping.csv (labels/splits)
        """
        super().__init__()
        path = os.path.abspath(path)
        self.mode = mode
        self.data, self.label, self.coords, self.patch_size_lv0, self.case_ids = [], [], [], [], []

        filenames = sorted(glob.glob(os.path.join(path, 'pt_files', 'conch_v1_5', '*.pt')))

        labels_path = os.path.join(path, 'camelyon16_official_split_mapping.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"CAMELYON16 labels CSV not found at {labels_path}")
        labels = pandas.read_csv(labels_path, index_col=0, header=None)

        for fname in filenames:
            npy = torch.load(fname)
            case_id = os.path.basename(fname).replace('.pt', '.tif')
            category = labels.loc[case_id][1]
            if mode == 'train' and category != 'training':
                continue

            logging.info('adding %s: %s', mode, fname)
            self.data.append(npy)
            pt_basename = os.path.basename(fname).replace('.pt', '')
            self.case_ids.append(pt_basename)

            # coords
            coord_path = os.path.join(path, 'h5_files', 'conch_v1_5', f'{pt_basename}.h5')
            try:
                with h5py.File(coord_path, 'r') as f:
                    coords = torch.from_numpy(f['coords'][:])
            except (OSError, KeyError, ValueError) as e:
                logging.warning("Failed to read coords from %s: %s", coord_path, e)
                coords = torch.empty((0, 2), dtype=torch.int64)
            self.coords.append(coords)

            self.patch_size_lv0.append(512)
            label = labels.loc[case_id][2]
            self.label.append(1 if label == 'tumor' else 0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.case_ids[index], self.data[index], self.label[index]
        return self.case_ids[index], self.data[index], self.label[index], self.coords[index], self.patch_size_lv0[index]
