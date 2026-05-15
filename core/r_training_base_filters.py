import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg

from scipy.sparse.linalg import norm

from core.arnoldi import (
    compare_fit_panelA,
    g_0,
    g_1,
    g_2,
    g_3,
    g_band_pass,
    g_band_rejection,
    g_high_pass,
    g_low_pass,
)


BASE_FILTERS = {
    "g_0": g_0,
    "g_1": g_1,
    "g_2": g_2,
    "g_3": g_3,
    "g_low_pass": g_low_pass,
    "g_high_pass": g_high_pass,
    "g_band_pass": g_band_pass,
    "g_band_rejection": g_band_rejection,
}


def _resolve_filter(filter_name):
    if callable(filter_name):
        return filter_name
    key = str(filter_name)
    if key not in BASE_FILTERS:
        raise ValueError(
            f"Unsupported base filter '{filter_name}'. "
            f"Available: {sorted(BASE_FILTERS)}"
        )
    return BASE_FILTERS[key]


def _apply_polynomial_filter(M, Y0, coeffs):
    coeffs = np.asarray(coeffs).reshape(-1)
    Yk = Y0.copy()
    Y_filtered = coeffs[0] * Yk
    for k in range(1, len(coeffs)):
        Yk = M @ Yk
        Y_filtered = Y_filtered + coeffs[k] * Yk
    return Y_filtered


def arnoldi_label_expansion(W, y_train, filter_name="g_0", degree=8):
    filter_fn = _resolve_filter(filter_name)

    W = W.copy().astype(np.float32)
    W = W + np.eye(W.shape[0])
    D = np.array(W.sum(1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ W @ D_inv_sqrt
    L = np.eye(W.shape[0]) - A_norm

    if filter_fn in [g_0, g_1, g_2, g_3]:
        coeffs = compare_fit_panelA(filter_fn, "Chebyshev", False, int(degree), -0.9, 0.9)
        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(A_norm, y_train.copy(), coeffs)

    elif filter_fn in [g_low_pass, g_high_pass, g_band_pass, g_band_rejection]:
        coeffs = compare_fit_panelA(filter_fn, "Chebyshev", False, int(degree), 0.0001, 2)
        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(L, y_train.copy(), coeffs)

    else:
        raise ValueError(f"Unsupported filter: {filter_name}")

    return Y_filtered


def absorption_probability_neumann(W, beta, k, return_dense):
    n = W.shape[0]
    print("Calculate absorption probability...")
    W = W.copy().astype(np.float32)
    D = W.sum(1).flat
    L = sp.diags(D, dtype=np.float32) - W
    L = sp.csc_matrix(L)

    I = sp.identity(n, format="csc")

    A = L + beta * I
    alpha = norm(A, np.inf)
    B = I - (A / alpha)

    S = I.copy()
    term = I.copy()
    for i in range(1, k + 1):
        term = term @ B
        S = S + term
        if i % 100 == 0:
            print(f"k= {i}")

    P = (1 / alpha) * S
    return P.toarray() if return_dense else P


def absorption_probability(W, alpha, stored_A=None, column=None):
    n = W.shape[0]
    print("Calculate absorption probability...")
    W = W.copy().astype(np.float32)
    D = W.sum(1).flat
    L = sp.diags(D, dtype=np.float32) - W
    L = sp.csc_matrix(L)

    I = sp.identity(n, format="csc")
    L = L + alpha * I

    if column is not None:
        A = np.zeros(W.shape)
        A[:, column] = slinalg.spsolve(
            L, sp.csc_matrix(np.eye(L.shape[0], dtype="float32")[:, column])
        ).toarray()
        return A

    A = slinalg.inv(L).toarray()
    if stored_A:
        np.savez(stored_A + str(alpha) + ".npz", A)
    return A


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


all_labels = None


def correct_label_count(indicator, i):
    if indicator.dtype == bool:
        total = np.where(indicator)[0].shape[0]
    elif indicator.dtype in [int, np.int8, np.int16, np.int32, np.int64]:
        total = indicator.shape[0]
    else:
        raise TypeError("indicator must be of data type np.bool or np.int")

    if all_labels is None:
        print("-", "/", total, sep="", end="\t")
        return
    count = np.sum(all_labels[:, i][indicator])
    print(count, "/", total, sep="", end="\t")


def r_training_label_expansion(W, t, y_train, train_mask, filter_name="g_0", degree=8):
    A = arnoldi_label_expansion(W, y_train, filter_name=filter_name, degree=degree)

    y_train = y_train.copy()
    n_nodes, n_classes = y_train.shape

    train_index = np.where(train_mask)[0]

    print("Additional Label:")

    r_training_idx_list = []
    r_training_label_list = []
    pos_in_added = {}

    best_score = np.full(n_nodes, -np.inf)
    best_label = np.full(n_nodes, -1)

    # Protect initial labels
    initial_idx = np.where(train_mask)[0]
    for idx in initial_idx:
        lbl = np.argmax(y_train[idx])
        best_label[idx] = lbl
        best_score[idx] = np.inf

    if not hasattr(t, "__getitem__"):
        t = [t for _ in range(n_classes)]

    for i in range(y_train.shape[1]):
        # y = y_train[:, i:i + 1]
        # a = A.dot(y)
        a = A[:, i:i + 1]

        a[initial_idx] = -np.inf

        # threshold-based
        # gate = (-np.sort(-a, axis=0))[t[i]]
        # index = np.where(a.flat > gate)[0]

        # exact count
        index = np.array(np.argsort(-a, axis=0).flatten())[0]
        index = index[: t[i]]

        for idx in index:
            new_score = a[idx, 0]

            if best_label[idx] == -1:
                best_label[idx] = i
                best_score[idx] = new_score

                y_train[idx] = 0
                y_train[idx, i] = 1

                pos = len(r_training_idx_list)
                pos_in_added[idx] = pos
                r_training_idx_list.append(idx)
                r_training_label_list.append(i)

                train_index = np.hstack([train_index, idx])

            else:
                old_score = best_score[idx]

                if new_score > old_score:
                    best_label[idx] = i
                    best_score[idx] = new_score

                    y_train[idx] = 0
                    y_train[idx, i] = 1

                    if idx in pos_in_added:
                        pos = pos_in_added[idx]
                        r_training_label_list[pos] = i

                else:
                    pass

        correct_label_count(index, i)

    print()
    # train_index = np.hstack([train_index, r_training_idx_list])
    train_mask = sample_mask(train_index, y_train.shape[0])
    return np.array(r_training_idx_list), np.array(r_training_label_list), y_train.copy(), train_mask.copy()


def compute_t_per_class(adj, y_train, dataset, tau=2, factor=3):
    n = adj.shape[0]
    avg_degree = adj.sum() / n
    eta = n / (avg_degree ** tau)
    labels_per_class = y_train.sum(axis=0)
    total_labels = labels_per_class.sum()

    t = (labels_per_class * factor * eta / total_labels).astype(np.int64)
    t[t < 1] = 1

    print(f"t per class for {dataset}: {t}")
    return t


def run_r_training_label_expansion(adj, labels, labeled_idx, dataset="", filter_name="g_0", degree=8):
    n = labels.shape[0]
    n_classes = len(np.unique(labels))

    global all_labels
    all_labels = np.eye(n_classes)[labels]

    y_train = np.zeros((n, n_classes))
    for idx in labeled_idx:
        y_train[idx, labels[idx]] = 1

    train_mask = sample_mask(labeled_idx, n)
    t = compute_t_per_class(adj, y_train, dataset)

    r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask = r_training_label_expansion(
        W=adj,
        t=t,
        y_train=y_train,
        train_mask=train_mask,
        filter_name=filter_name,
        degree=degree,
    )
    return r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask
