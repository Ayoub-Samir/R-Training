# # core/sample_labels.py
# import numpy as np

# # Configuration: number of labeled nodes per class for each dataset
# LABELS_PER_CLASS = {
#     'texas': 2,
#     'cornell': 2,
#     'chameleon': 25,
#     'squirrel': 50
# }

# def sample_initial_labels(labels, dataset):
#     """
#     Randomly select a fixed number of labeled nodes per class.

#     Args:
#         labels   : numpy array of shape [n_nodes]
#         dataset  : dataset name ('texas', 'cornell', 'chameleon', 'squirrel')

#     Returns:
#         idx_labeled : numpy array of selected node indices
#     """
#     dataset = dataset.lower()
#     if dataset not in LABELS_PER_CLASS:
#         raise ValueError(f"Dataset {dataset} not in LABELS_PER_CLASS config")

#     num_per_class = LABELS_PER_CLASS[dataset]
#     classes = np.unique(labels)

#     labeled_indices = []
#     np.random.seed(42)  # for reproducibility

#     for c in classes:
#         idx = np.where(labels == c)[0]
#         if len(idx) == 1:
#             chosen = np.random.choice(idx, 1, replace=False)
#             labeled_indices.extend(chosen)
#         else:
#             if len(idx) < num_per_class:
#                 raise ValueError(f"Class {c} in {dataset} has only {len(idx)} nodes")
#             chosen = np.random.choice(idx, num_per_class, replace=False)
#             labeled_indices.extend(chosen)

#     labeled_indices = np.array(sorted(labeled_indices))

#     print(f"{dataset.upper()}: selected {num_per_class} labeled nodes per class "
#           f"({len(labeled_indices)} total)")
#     return labeled_indices



import json
from pathlib import Path

import numpy as np

def sample_initial_labels(labels, dataset, label_rate=0.05, random_seed=42):

    np.random.seed(random_seed)
    classes = np.unique(labels)
    labeled_indices = []

    for c in classes:
        idx = np.where(labels == c)[0]
        num_to_label = max(1, int(np.ceil(len(idx) * label_rate)))
        chosen = np.random.choice(idx, num_to_label, replace=False)
        labeled_indices.extend(chosen)

    labeled_indices = np.array(sorted(labeled_indices))
    print(f"{dataset.upper()}: selected {label_rate*100:.1f}% per class "
          f"({len(labeled_indices)} total labeled nodes)")
    return labeled_indices


# def sample_initial_labels(dataset, k, splits_dir="splits"):
#     dataset = dataset.lower()

#     try:
#         k = int(k)
#     except (TypeError, ValueError) as exc:
#         raise ValueError("k must be one of {1, 2, 4, 8, 16, 20}.") from exc

#     allowed = {1, 2, 4, 8, 16, 20}
#     if k not in allowed:
#         raise ValueError(f"k must be one of {sorted(allowed)}.")

#     split_path = Path(splits_dir) / f"ind.{dataset}_l{k}.split.json"
#     if not split_path.is_file():
#         raise FileNotFoundError(f"Split file not found: {split_path}")

#     with split_path.open("r", encoding="utf-8") as handle:
#         split = json.load(handle)

#     if "train_indices" not in split:
#         raise KeyError(f"'train_indices' not found in {split_path}")

#     labeled_indices = np.array(sorted(split["train_indices"]), dtype=np.int64)
#     print(f"{dataset.upper()}: loaded {len(labeled_indices)} labeled nodes from {split_path}")
#     return labeled_indices
