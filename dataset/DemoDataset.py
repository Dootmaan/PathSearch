"""Demo Dataset for PathSearch testing.

This module provides a simple dataset class that reads pre-extracted
feature tensors (.pt files) from the demo_dataset folder for quick
demonstration of the PathSearch retrieval system.
"""
import os
import torch
import logging
import pandas as pd
from typing import Optional, List, Tuple, Dict
from fast_pytorch_kmeans import KMeans


# Label mapping consistent with TCGATestingDataset
LABEL_MAP = {
    'LUSC': 0,
    'LUAD': 1,
    'KICH': 2,
    'KIRC': 3,
    'KIRP': 4,
    'IDC': 5,
    'ILC': 6,
}

# Reverse mapping for display
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_tcga_labels() -> Dict[str, str]:
    """Load all TCGA labels from CSV files.
    
    Returns:
        Dictionary mapping case_id (first 12 chars) to label string (e.g., 'LUAD', 'IDC', etc.)
    """
    labels = {}
    
    # Label CSV files (try multiple possible locations)
    label_files = [
        # Relative to project root
        './data/TCGA/labels/BRCA_subtyping.csv',
        './data/TCGA/labels/LUAD_LUSC.csv',
        './data/TCGA/labels/RCC.csv',
        # Absolute paths (fallback)
        '/home/hongyi/Workspace-Python/EasyMIL/dataset_csv/BRCA_subtyping.csv',
        '/home/hongyi/Workspace-Python/EasyMIL/dataset_csv/LUAD_LUSC.csv',
        '/home/hongyi/Workspace-Python/EasyMIL/dataset_csv/RCC.csv',
    ]
    
    for csv_path in label_files:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'slide_id' in df.columns and 'label' in df.columns:
                    for _, row in df.iterrows():
                        slide_id = row['slide_id']
                        label = row['label']
                        # Store both full slide_id and case_id (first 12 chars)
                        labels[slide_id] = label
                        case_id = slide_id[:12]
                        labels[case_id] = label
            except Exception as e:
                logging.warning(f"Failed to load {csv_path}: {e}")
    
    return labels


class DemoDataset(torch.utils.data.Dataset):
    """Dataset for loading demo .pt files for PathSearch testing.

    Expected layout:
      ./demo_dataset/<CASE>.pt   (each .pt contains patch-level features of shape (N, 768))

    Labels are loaded from TCGA label CSV files, supporting 7 classes:
    LUSC (0), LUAD (1), KICH (2), KIRC (3), KIRP (4), IDC (5), ILC (6)
    """

    def __init__(
        self,
        path: str = './demo_dataset',
        mode: str = 'test',
        sample_num: Optional[int] = None,
    ):
        """Initialize the demo dataset.

        Args:
            path: Path to the demo_dataset folder containing .pt files.
            mode: Dataset mode (train/test). For demo purposes, only 'test' is used.
            sample_num: If specified, use KMeans to sample this many patches.
        """
        super().__init__()
        self.mode = mode
        self.sample_num = sample_num
        self.kmeans = KMeans(n_clusters=sample_num, mode='euclidean', verbose=0) if sample_num is not None else None

        self.data: List[torch.Tensor] = []
        self.label: List[int] = []
        self.case_id: List[str] = []
        self.coords: List[torch.Tensor] = []
        self.patch_size_lv0: List[int] = []

        # Load all TCGA labels from CSV files
        tcga_labels = load_tcga_labels()
        logging.info(f"Loaded {len(tcga_labels)} TCGA labels from CSV files")

        # Collect all .pt files in the demo_dataset folder
        if not os.path.exists(path):
            raise FileNotFoundError(f"Demo dataset path not found: {path}")

        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        if len(pt_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {path}")

        logging.info(f"Found {len(pt_files)} demo files in {path}")

        # Load each .pt file
        label_counts = {name: 0 for name in LABEL_MAP.keys()}
        for pt_file in pt_files:
            case = pt_file.replace('.pt', '')
            feat_path = os.path.join(path, pt_file)

            try:
                npy = torch.load(feat_path, weights_only=True)
                if len(npy) == 0:
                    logging.warning(f"Empty features in {feat_path}, skipping.")
                    continue
            except (OSError, RuntimeError) as e:
                logging.warning(f"Failed to load {feat_path}: {e}")
                continue

            self.data.append(npy)
            self.case_id.append(case)
            
            # Look up label from TCGA labels
            if case in tcga_labels:
                label_str = tcga_labels[case]
                label_int = LABEL_MAP.get(label_str, -1)
                self.label.append(label_int)
                if label_str in label_counts:
                    label_counts[label_str] += 1
            else:
                logging.warning(f"Unknown case {case}, assigning label -1")
                self.label.append(-1)
            
            # Placeholder coords (not available for demo)
            self.coords.append(torch.zeros((npy.shape[0], 2), dtype=torch.int64))
            self.patch_size_lv0.append(512)

        # Log label distribution
        logging.info(f"Loaded {len(self.data)} demo cases with label distribution: {label_counts}")

        logging.info(f"Loaded {len(self.data)} demo cases.")

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, int, torch.Tensor, int]:
        """Get a demo sample.

        Returns:
            (case_id, features, label, coords, patch_size)
        """
        data = self.data[index]

        if self.sample_num is not None and len(data) > self.sample_num:
            _ = self.kmeans.fit_predict(data)
            data = self.kmeans.centroids
        elif self.sample_num is not None:
            # Pad if fewer patches than sample_num
            import random
            data = torch.tensor(random.choices(data.tolist(), k=self.sample_num))

        return (
            self.case_id[index],
            data,
            self.label[index],
            self.coords[index],
            self.patch_size_lv0[index],
        )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    dataset = DemoDataset('./demo_dataset', mode='test')
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        case_id, data, label, coords, patch_size = dataset[0]
        print(f"Sample - Case: {case_id}, Features shape: {data.shape}, Label: {label}")
