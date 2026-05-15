import numpy as np
import torch
import scipy.sparse as sp
import json
import io
import contextlib
from pathlib import Path
from functools import lru_cache

from dataset_utils import DataLoader
from core.sample_labels import sample_initial_labels
from core.r_training_adaptive_filters import ADAPTIVE_FILTER_DEFAULT_CFG, run_r_training_label_expansion as run_adaptive_r_training
from core.r_training_base_filters import run_r_training_label_expansion as run_base_r_training
from core.evaluate_r_training_labels import evaluate_r_training_labels

from core.gcn_model import get_model_class
from core.train_gcn import train_and_evaluate_gcn


class GCNConfig:
    model_type = "gcn"
    hidden = 64
    dropout = 0.0
    lr = 0.01
    weight_decay = 1e-3
    epochs = 200
    print_every = 20


HOMOPHILIC_DATASETS = {'cora', 'citeseer', 'pubmed', 'photo', 'computers'}
BINARY_HETEROPHILIC_DATASETS = {'minesweeper', 'tolokers', 'questions'}
MULTICLASS_HETEROPHILIC_DATASETS = {'texas', 'cornell', 'actor', 'chameleon', 'squirrel'}

ADAPTIVE_FILTER_DATASETS = MULTICLASS_HETEROPHILIC_DATASETS

datasets = [
    'cora', 'citeseer', 'pubmed', 'photo', 'computers',
    'texas', 'cornell', 'actor',
    'chameleon', 'squirrel',
    'minesweeper', 'tolokers', 'questions',
]

label_rates = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
num_runs = 10
train_seed = 0

ADAPTIVE_FILTER_PARAMS_DIR = Path("tuned_params")
BASE_FILTER_SAMPLE_SEED = 42
BASE_FILTER_NAME = "g_0"
BASE_FILTER_DEGREE = 8

ADAPTIVE_FILTER_GCN_CFG = {
    "model_type": "gcn",
    "hidden": 64,
    "dropout": 0.0,
    "lr": 0.01,
    "weight_decay": 1e-3,
    "epochs": 200,
}

BASE_FILTER_GCN_CFG = {
    "model_type": "gcn",
    "hidden": 64,
    "dropout": 0.5,
    "lr": 0.002,
    "weight_decay": 5e-4,
    "epochs": 200,
}

MODEL_CFG_KEYS = (
    "model_type",
    "hidden",
    "dropout",
    "lr",
    "weight_decay",
    "epochs",
)


def _rate_key(label_rate):
    return f"{float(label_rate):.3f}"


def _adaptive_params_path(dataset_name):
    return ADAPTIVE_FILTER_PARAMS_DIR / f"{dataset_name.lower()}.json"


@lru_cache(maxsize=None)
def load_adaptive_filter_params(dataset_name):
    path = _adaptive_params_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(f"Adaptive filter params not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_adaptive_filter_entry(dataset_name, label_rate):
    target = float(label_rate)
    for entry in load_adaptive_filter_params(dataset_name):
        if abs(float(entry["label_rate"]) - target) < 1e-12:
            return entry
    raise KeyError(f"No adaptive filter params for {dataset_name} label_rate={label_rate}")


def get_adaptive_filter_sample_seed(dataset_name, label_rate):
    return int(get_adaptive_filter_entry(dataset_name, label_rate)["sample_seed"])


def get_adaptive_r_training_filter_cfg(dataset_name, label_rate):
    return dict(get_adaptive_filter_entry(dataset_name, label_rate)["r_training_filter_cfg"])


def get_r_training_branch(dataset_name):
    dataset_key = dataset_name.lower()
    if dataset_key in ADAPTIVE_FILTER_DATASETS:
        return "adaptive_filters"
    if dataset_key in HOMOPHILIC_DATASETS or dataset_key in BINARY_HETEROPHILIC_DATASETS:
        return "base_filters"
    raise ValueError(
        f"Dataset '{dataset_name}' is not configured. "
        f"Adaptive-filter datasets: {sorted(ADAPTIVE_FILTER_DATASETS)}; homophilic datasets: {sorted(HOMOPHILIC_DATASETS)}; binary heterophilic datasets: {sorted(BINARY_HETEROPHILIC_DATASETS)}"
    )


def get_default_model_cfg(r_training_branch):
    if r_training_branch == "adaptive_filters":
        return dict(ADAPTIVE_FILTER_GCN_CFG)
    if r_training_branch == "base_filters":
        return dict(BASE_FILTER_GCN_CFG)
    raise ValueError(f"Unknown r_training_branch: {r_training_branch}")


def get_sample_seed(dataset_name, label_rate, r_training_branch):
    if r_training_branch == "adaptive_filters":
        return get_adaptive_filter_sample_seed(dataset_name, label_rate)
    if r_training_branch == "base_filters":
        return BASE_FILTER_SAMPLE_SEED
    raise ValueError(f"Unknown r_training_branch: {r_training_branch}")


def build_args_from_cfg(cfg):
    args = GCNConfig()
    for key in MODEL_CFG_KEYS:
        if key in cfg:
            setattr(args, key, cfg[key])
    return args


def extract_model_cfg(cfg):
    return {key: cfg[key] for key in MODEL_CFG_KEYS if key in cfg}


@lru_cache(maxsize=None)
def load_dataset_payload(dataset_name):
    loader_name = 'film' if dataset_name == 'actor' else dataset_name
    _, data = DataLoader(loader_name)

    edge_index = data.edge_index.cpu().numpy()
    n_nodes = data.x.shape[0]
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(n_nodes, n_nodes),
        dtype=float
    )

    return {
        "data": data,
        "adj": adj,
        "features": data.x.cpu().numpy(),
        "labels": data.y.cpu().numpy(),
    }


def run_one_config(
    dataset_name,
    label_rate,
    model_cfg=None,
    r_training_filter_cfg=None,
    num_runs_override=None,
    verbose=True,
    evaluate_new_labels=True,
    use_r_training=True,
):
    r_training_branch = get_r_training_branch(dataset_name)
    cfg = get_default_model_cfg(r_training_branch)
    if model_cfg is not None:
        cfg.update(model_cfg)
    cfg = extract_model_cfg(cfg)
    sample_seed = int(
        model_cfg.get("sample_seed", get_sample_seed(dataset_name, label_rate, r_training_branch))
        if model_cfg is not None
        else get_sample_seed(dataset_name, label_rate, r_training_branch)
    )

    if r_training_branch == "adaptive_filters":
        r_training_cfg = dict(ADAPTIVE_FILTER_DEFAULT_CFG)
        r_training_cfg.update(get_adaptive_r_training_filter_cfg(dataset_name, label_rate))
        if r_training_filter_cfg is not None:
            r_training_cfg.update(r_training_filter_cfg)
    else:
        r_training_cfg = {
            "filter_name": BASE_FILTER_NAME,
            "degree": BASE_FILTER_DEGREE,
        }
        if r_training_filter_cfg is not None:
            r_training_cfg.update(r_training_filter_cfg)

    payload = load_dataset_payload(dataset_name)
    data = payload["data"].clone()
    adj = payload["adj"]
    features = payload["features"]
    labels = payload["labels"]
    data.true_y = torch.tensor(labels, dtype=torch.long).clone()

    if verbose:
        idx_labeled = sample_initial_labels(
            labels,
            dataset_name,
            label_rate=label_rate,
            random_seed=sample_seed,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            idx_labeled = sample_initial_labels(
                labels,
                dataset_name,
                label_rate=label_rate,
                random_seed=sample_seed,
            )

    if use_r_training:
        if r_training_branch == "adaptive_filters":
            run_r_training_fn = run_adaptive_r_training
            run_r_training_kwargs = {
                "adj": adj,
                "labels": labels,
                "labeled_idx": idx_labeled,
                "dataset": dataset_name,
                "r_training_cfg": r_training_cfg,
            }
        else:
            run_r_training_fn = run_base_r_training
            run_r_training_kwargs = {
                "adj": adj,
                "labels": labels,
                "labeled_idx": idx_labeled,
                "dataset": dataset_name,
                "filter_name": r_training_cfg["filter_name"],
                "degree": r_training_cfg["degree"],
            }

        if verbose:
            r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask = run_r_training_fn(**run_r_training_kwargs)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask = run_r_training_fn(**run_r_training_kwargs)
    else:
        r_training_idx = np.asarray([], dtype=np.int64)
        r_training_pred = np.asarray([], dtype=np.int64)
        num_classes = len(np.unique(labels))
        expanded_label_matrix = np.zeros((len(labels), num_classes), dtype=np.float32)
        expanded_label_matrix[idx_labeled, labels[idx_labeled]] = 1.0
        expanded_train_mask = np.zeros(len(labels), dtype=bool)
        expanded_train_mask[idx_labeled] = True

    r_training_label_acc = None
    if len(r_training_idx) > 0:
        r_training_label_acc = float(np.mean(labels[r_training_idx] == r_training_pred))
        if evaluate_new_labels and verbose:
            evaluate_r_training_labels(labels, r_training_idx, r_training_pred, dataset=dataset_name)

    expanded_y = np.argmax(expanded_label_matrix, axis=1)
    unlabeled_mask = (expanded_label_matrix.sum(axis=1) == 0)
    expanded_y[unlabeled_mask] = -1
    data.y = torch.tensor(expanded_y, dtype=torch.long)
    data.train_mask = torch.tensor(expanded_train_mask, dtype=torch.bool)

    class DatasetInfo:
        num_features = features.shape[1]
        num_classes = len(np.unique(labels))
        num_nodes = features.shape[0]

    dataset_info = DatasetInfo()
    args = build_args_from_cfg(cfg)
    model_type = str(cfg.get("model_type", "gcn")).lower()
    Net = get_model_class(model_type)
    run_count = num_runs if num_runs_override is None else num_runs_override

    accuracies = []
    f1s = []
    roc_aucs = []
    cm = None
    for run_idx in range(run_count):
        seed = train_seed + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        if verbose:
            accuracy, f1, roc_auc, cm, model = train_and_evaluate_gcn(
                args=args,
                dataset=dataset_info,
                data=data,
                Net=Net
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                accuracy, f1, roc_auc, cm, model = train_and_evaluate_gcn(
                    args=args,
                    dataset=dataset_info,
                    data=data,
                    Net=Net
                )
        accuracies.append(float(accuracy))
        f1s.append(float(f1))
        if roc_auc is not None:
            roc_aucs.append(float(roc_auc))

    acc_mean = float(np.mean(accuracies)) * 100.0
    acc_std = float(np.std(accuracies, ddof=1)) * 100.0 if len(accuracies) > 1 else 0.0

    return {
        "dataset": dataset_name,
        "label_rate": float(label_rate),
        "r_training_branch": r_training_branch,
        "model_type": model_type,
        "use_r_training": bool(use_r_training),
        "model_cfg": cfg,
        "sample_seed": sample_seed,
        "train_seed": train_seed,
        "r_training_filter_cfg": r_training_cfg,
        "initial_labels": int(len(idx_labeled)),
        "r_training_added_labels": int(len(r_training_idx)),
        "r_training_idx": np.asarray(r_training_idx).astype(np.int64).tolist(),
        "r_training_pred": np.asarray(r_training_pred).astype(np.int64).tolist(),
        "r_training_label_acc": r_training_label_acc,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "roc_auc_mean": float(np.mean(roc_aucs)) if len(roc_aucs) > 0 else None,
        "roc_auc_std": float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0 if len(roc_aucs) == 1 else None,
        "cm_last": cm.tolist() if cm is not None else None,
    }


def main():
    for dataset_name in datasets:
        print("\n===========================================")
        print(f"Running Pipeline on {dataset_name.upper()}")
        print("===========================================\n")

        rate_results = []

        for label_rate in label_rates:
            result = run_one_config(
                dataset_name=dataset_name,
                label_rate=label_rate,
                model_cfg=None,
                r_training_filter_cfg=None,
                num_runs_override=num_runs,
                verbose=True,
                evaluate_new_labels=True,
            )

            print(f"\n\nLabel Rate: {label_rate}")
            print(f"Initial Labels: {result['initial_labels']}")
            print(f"R-Training Added Labels: {result['r_training_added_labels']}")
            print("-----------------------------------")

            print("\n========== FINAL RESULTS ==========")
            print(f"Dataset      : {dataset_name}")
            print(f"Label Rate   : {label_rate}")
            print(f"Accuracy     : {result['acc_mean']:.2f}+-{result['acc_std']:.2f}")
            print(f"Macro-F1     : {result['macro_f1_mean']:.4f}")
            print("Confusion Matrix (last run):")
            print(np.array(result["cm_last"]))
            print("====================================\n")

            rate_results.append((label_rate, result["acc_mean"], result["acc_std"]))

        print("\n========== RATE SUMMARY ==========")
        for lr, m, s in rate_results:
            print(f"label_rate={lr:.3f} -> {m:.2f}+-{s:.2f}")
        print("==================================\n")


if __name__ == "__main__":
    main()
