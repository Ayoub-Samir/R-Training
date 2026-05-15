import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
import math

from scipy.sparse.linalg import norm

from core.arnoldi import *


ADAPTIVE_FILTER_DEFAULT_CFG = {
    "hetero_w0": 0.2,
    "hetero_w1": 0.5,
    "hetero_w2": 0.3,
    "hetero_g0": 6.0,
    "hetero_g1": 10.0,
    "hetero_g2": 6.0,
    "hetero_degree": 6,
    "beta_filter": "beta_fit",
    "beta_p": 2,
    "beta_q": 2,
    "beta_degree": 6,
    "beta_C": 2,
    "beta_agg": "mean",
    "mix_het": 0.6,
    "mix_beta": 0.4,
    "budget_tau": 2.0,
    "budget_factor": 2.0,
}


def l2_normalize(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return X / norm


def _resolve_r_training_cfg(cfg=None):
    resolved = dict(ADAPTIVE_FILTER_DEFAULT_CFG)
    if cfg is not None:
        resolved.update(cfg)
    return resolved


def _hetero_filter_kernel(x, cfg):
    return (
        cfg["hetero_w0"] * np.exp(-cfg["hetero_g0"] * x**2)
        + cfg["hetero_w1"] * np.exp(-cfg["hetero_g1"] * (x - 1.0) ** 2)
        + cfg["hetero_w2"] * np.exp(-cfg["hetero_g2"] * (x - 2.0) ** 2)
    )


def feature_similarity_score(W, X):
    Xn = l2_normalize(X)
    sim = Xn @ Xn.T
    H_feat = (W * sim).sum(axis=1) / (W.sum(axis=1) + 1e-8)
    H_feat = np.asarray(H_feat).reshape(-1)
    return H_feat, sim


def neighborhood_overlap_score(W, sim, k=10):
    n = W.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    k = max(1, min(int(k), n - 1))
    sim_knn = sim.copy()
    np.fill_diagonal(sim_knn, -np.inf)
    knn_idx = np.argpartition(sim_knn, -k, axis=1)[:, -k:]

    H_overlap = np.zeros(n, dtype=np.float32)
    for i in range(n):
        g = set(np.where(W[i] > 0)[0].tolist())
        if i in g:
            g.remove(i)
        f = set(knn_idx[i].tolist())

        union = g | f
        inter = g & f
        H_overlap[i] = len(inter) / (len(union) + 1e-8) if len(union) > 0 else 0.0

    return H_overlap


def stability_score(W, sim, num_runs=5, drop_rate=0.1, seed=42):
    n = W.shape[0]
    if num_runs <= 1 or drop_rate <= 0:
        return np.ones(n, dtype=np.float32)

    ei, ej = np.where(W > 0)
    ecount = ei.shape[0]
    if ecount == 0:
        return np.zeros(n, dtype=np.float32)

    rng = np.random.default_rng(seed)
    hist = np.zeros((num_runs, n), dtype=np.float32)

    for r in range(num_runs):
        keep = rng.random(ecount) >= drop_rate
        if keep.sum() == 0:
            keep[rng.integers(0, ecount)] = True

        W_drop = np.zeros_like(W, dtype=np.float32)
        W_drop[ei[keep], ej[keep]] = W[ei[keep], ej[keep]]

        h = (W_drop * sim).sum(axis=1) / (W_drop.sum(axis=1) + 1e-8)
        hist[r] = np.asarray(h).reshape(-1)

    mu = hist.mean(axis=0)
    sigma = hist.std(axis=0)
    H_stability = 1.0 - (sigma / (np.abs(mu) + 1e-8))
    return np.clip(H_stability, 0.0, 1.0)


def _minmax01(v):
    v = np.asarray(v, dtype=np.float32)
    lo = float(v.min())
    hi = float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def local_homophily_gate(W, X, top_percent=0.25):
    W = W.copy().astype(np.float32).toarray()

    # 1) feature similarity score (same formula as your original code)
    H_feat, sim = feature_similarity_score(W, X)

    # 2) graph-neighbor vs feature-kNN overlap score
    H_overlap = neighborhood_overlap_score(W, sim, k=10)

    # # 3) stability score from random edge-dropout perturbations
    # H_stability = stability_score(W, sim, num_runs=5, drop_rate=0.10, seed=42)

    # final proxy score
    H_local = (
        0.9 * _minmax01(H_feat)
        + 0.1 * _minmax01(H_overlap)
    )
    H_local = np.asarray(H_local).reshape(-1)
    
    top_k = max(1, int(np.ceil(top_percent * H_local.shape[0])))
    top_idx = np.argsort(-H_local)[:top_k].astype(np.int64)
    return top_idx


def beta_wavelet_kernel(x, p=1, q=1):
    p = int(p)
    q = int(q)
    if p < 0 or q < 0:
        raise ValueError("beta_wavelet_kernel requires p, q >= 0.")

    # B(p+1, q+1) for integer p, q.
    beta_norm = (
        math.factorial(p) * math.factorial(q) / math.factorial(p + q + 1)
    )
    return ((x / 2.0) ** p) * ((1.0 - x / 2.0) ** q) / (2.0 * beta_norm)


def beta_wavelet_exact_coeffs(p=1, q=1):
    p = int(p)
    q = int(q)
    if p < 0 or q < 0:
        raise ValueError("beta_wavelet_exact_coeffs requires p, q >= 0.")

    deg = p + q
    coeffs = np.zeros(deg + 1, dtype=np.float64)

    beta_norm = (
        math.factorial(p) * math.factorial(q) / math.factorial(deg + 1)
    )
    scale = 1.0 / (2.0 * beta_norm)

    # g(lam) = scale * (lam/2)^p * (1 - lam/2)^q
    #        = sum_{j=0..q} scale * C(q,j) * (-1)^j * lam^(p+j) / 2^(p+j)
    for j in range(q + 1):
        k = p + j
        coeffs[k] += (
            scale
            * math.comb(q, j)
            * ((-1.0) ** j)
            / (2.0 ** k)
        )
    return coeffs


def _apply_polynomial_filter(M, Y0, coeffs):
    coeffs = np.asarray(coeffs).reshape(-1)
    Yk = Y0.copy()
    Y_filtered = coeffs[0] * Yk
    for k in range(1, len(coeffs)):
        Yk = M @ Yk
        Y_filtered = Y_filtered + coeffs[k] * Yk
    return Y_filtered


def _aggregate_channel_outputs(outputs, mode="mean"):
    if len(outputs) == 0:
        raise ValueError("No channel outputs to aggregate.")
    if mode == "mean":
        return np.mean(outputs, axis=0)
    if mode == "sum":
        return np.sum(outputs, axis=0)
    raise ValueError(f"Unsupported beta_agg mode: {mode}. Use 'mean' or 'sum'.")


def arnoldi_label_expansion_beta_exact(W, y_train, p=1, q=1):
    return arnoldi_label_expansion(
        W,
        y_train,
        filter="beta_exact",
        beta_p=p,
        beta_q=q,
    )


def arnoldi_label_expansion_beta_fit(W, y_train, p=2, q=2, degree=6):
    return arnoldi_label_expansion(
        W,
        y_train,
        filter="beta_fit",
        beta_p=p,
        beta_q=q,
        beta_degree=degree,
    )


def arnoldi_label_expansion_beta_multi_exact(W, y_train, C=2, agg="mean"):
    return arnoldi_label_expansion(
        W,
        y_train,
        filter="beta_multi_exact",
        beta_C=C,
        beta_agg=agg,
    )


def arnoldi_label_expansion_beta_multi_fit(W, y_train, C=2, degree=10, agg="mean"):
    return arnoldi_label_expansion(
        W,
        y_train,
        filter="beta_multi_fit",
        beta_C=C,
        beta_degree=degree,
        beta_agg=agg,
    )



def arnoldi_label_expansion(
    W,
    y_train,
    filter=g_hetero,
    beta_p=2,
    beta_q=2,
    degree=6,
    beta_degree=6,
    beta_C=2,
    beta_agg="mean",
    r_training_cfg=None,
):
    r_training_cfg = _resolve_r_training_cfg(r_training_cfg)
    W = W.copy().astype(np.float32)
    W = W + np.eye(W.shape[0])  # add self-loops
    D = np.array(W.sum(1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D)) # D + 1e-8
    A_norm = D_inv_sqrt @ W @ D_inv_sqrt
    L = np.eye(W.shape[0]) - A_norm

    if filter == "hetero_custom":
        coeffs = compare_fit_panelA(
            lambda x, cfg=r_training_cfg: _hetero_filter_kernel(x, cfg),
            "Chebyshev",
            False,
            int(degree),
            0.0001,
            2,
        )
        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(L, y_train.copy(), coeffs)

    elif filter == "beta_exact":
        coeffs = beta_wavelet_exact_coeffs(beta_p, beta_q)
        Y_filtered = _apply_polynomial_filter(L, y_train.copy(), coeffs)

    elif filter == "beta_fit":
        coeffs = compare_fit_panelA(
            lambda x: beta_wavelet_kernel(x, beta_p, beta_q),
            "Chebyshev",
            False,
            int(beta_degree),
            0.0001,
            2,
        )
        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(L, y_train.copy(), coeffs)

    elif filter == "beta_multi_exact":
        C = int(beta_C)
        if C < 0:
            raise ValueError("beta_C must be >= 0.")

        outputs = []
        for p in range(C + 1):
            q = C - p
            coeffs = beta_wavelet_exact_coeffs(p, q)
            outputs.append(_apply_polynomial_filter(L, y_train.copy(), coeffs))
        Y_filtered = _aggregate_channel_outputs(outputs, mode=beta_agg)

    elif filter == "beta_multi_fit":
        C = int(beta_C)
        if C < 0:
            raise ValueError("beta_C must be >= 0.")

        outputs = []
        for p in range(C + 1):
            q = C - p
            coeffs = compare_fit_panelA(
                lambda x, pp=p, qq=q: beta_wavelet_kernel(x, pp, qq),
                "Chebyshev",
                False,
                int(beta_degree),
                0.0001,
                2,
            )
            coeffs = coeffs[::-1]
            outputs.append(_apply_polynomial_filter(L, y_train.copy(), coeffs))
        Y_filtered = _aggregate_channel_outputs(outputs, mode=beta_agg)


    elif filter in [g_0, g_1, g_2, g_3, g_hetero_adj]:
        coeffs = compare_fit_panelA(filter, 'Chebyshev', False, int(degree), -0.9, 0.9)

        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(A_norm, y_train.copy(), coeffs)
            
    elif filter in [g_low_pass, g_high_pass, g_band_pass, g_band_rejection, g_hetero]:
        coeffs = compare_fit_panelA(filter, 'Chebyshev', False, int(degree), 0.0001, 2)

        coeffs = coeffs[::-1]
        Y_filtered = _apply_polynomial_filter(L, y_train.copy(), coeffs)
    else:
        raise ValueError(f"Unsupported filter: {filter}")
    

    # # coeffs = coeffs[::-1]
    # Y_filtered = coeffs[-1] * Y
    # for k in range(len(coeffs)-2, -1, -1):
    #     Y = A_norm @ Y
    #     Y_filtered = coeffs[k] * Y + Y_filtered

    return Y_filtered




def absorption_probability_neumann(W, beta, k, return_dense):
    n = W.shape[0]
    print('Calculate absorption probability...')
    W = W.copy().astype(np.float32)
    D = W.sum(1).flat
    L = sp.diags(D, dtype=np.float32) - W
    L = sp.csc_matrix(L)

    I = sp.identity(n, format="csc")
    
    A = L + beta * I
    alpha = norm(A, np.inf)
    # alpha = np.max(np.abs(A).sum(axis=1))
    # alpha = float(alpha)

    B = I - (A / alpha)
    
    S =  I.copy()
    term = I.copy()
    for _ in range(1, k+1):
        term = term @ B
        S = S + term
        if _ % 100 == 0: print(f"k= {_}")

    P = (1 / alpha) * S
    
    return P.toarray() if return_dense else P     

def absorption_probability(W, alpha, stored_A=None, column=None):

    n = W.shape[0]
    print('Calculate absorption probability...')
    W = W.copy().astype(np.float32)
    D = W.sum(1).flat
    L = sp.diags(D, dtype=np.float32) - W
    L = sp.csc_matrix(L)
    
    I = sp.identity(n, format="csc")
    L = L + alpha * I

    if column is not None:
        A = np.zeros(W.shape)
        A[:, column] = slinalg.spsolve(
            L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])
        ).toarray()
        return A
    else:
        A = slinalg.inv(L).toarray()
        if stored_A:
            np.savez(stored_A + str(alpha) + '.npz', A)
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
        raise TypeError('indicator must be of data type np.bool or np.int')
    # Keep diagnostics leakage-free unless explicitly enabled.
    if all_labels is None:
        print('-', '/', total, sep='', end='\t')
        return
    count = np.sum(all_labels[:, i][indicator])
    print(count, '/', total, sep='', end='\t')


def _column_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    mu = scores.mean(axis=0, keepdims=True)
    sigma = scores.std(axis=0, keepdims=True) + 1e-8
    return (scores - mu) / sigma



def r_training_label_expansion(W, t, y_train, train_mask, r_training_cfg=None):
    r_training_cfg = _resolve_r_training_cfg(r_training_cfg)
    y_train = y_train.copy()
    n_nodes, n_classes = y_train.shape

    initial_idx = np.where(train_mask)[0]
    initial_set = set(initial_idx.tolist())
    train_index = initial_idx.tolist()

    if not hasattr(t, "__getitem__"):
        t = np.array([t for _ in range(n_classes)], dtype=np.int64)
    else:
        t = np.asarray(t, dtype=np.int64)

    # Two-filter scoring, then normalized mixing.
    A_het = np.asarray(
        arnoldi_label_expansion(
            W,
            y_train,
            filter="hetero_custom",
            degree=int(r_training_cfg["hetero_degree"]),
            r_training_cfg=r_training_cfg,
        )
    )
    A_beta = np.asarray(
        arnoldi_label_expansion(
            W,
            y_train,
            filter=r_training_cfg["beta_filter"],
            beta_p=int(r_training_cfg["beta_p"]),
            beta_q=int(r_training_cfg["beta_q"]),
            beta_degree=int(r_training_cfg["beta_degree"]),
            beta_C=int(r_training_cfg["beta_C"]),
            beta_agg=r_training_cfg["beta_agg"],
            r_training_cfg=r_training_cfg,
        )
    )
    S_het = _column_zscore(A_het)
    S_beta = _column_zscore(A_beta)
    S_mix = (
        float(r_training_cfg["mix_het"]) * S_het
        + float(r_training_cfg["mix_beta"]) * S_beta
    )

    # R-Training label assignment:
    # predict with the mixed score only, then rank by mixed-score margin.
    class_candidates = [[] for _ in range(n_classes)]
    for idx in range(n_nodes):
        if idx in initial_set:
            continue

        top_mix = np.argsort(-S_mix[idx])[:2]
        c = int(top_mix[0])
        if n_classes > 1:
            conf = float(S_mix[idx, top_mix[0]] - S_mix[idx, top_mix[1]])
        else:
            conf = float(S_mix[idx, top_mix[0]])
        class_candidates[c].append((idx, conf))

    print("Additional Label:")
    r_training_idx_list = []
    r_training_label_list = []

    # Per-class top-t pick.
    for c in range(n_classes):
        cands = class_candidates[c]
        if len(cands) == 0:
            correct_label_count(np.array([], dtype=np.int64), c)
            continue

        cands.sort(key=lambda x: x[1], reverse=True)
        picked = cands[: int(t[c])]
        selected_idx = np.array([x[0] for x in picked], dtype=np.int64)

        if selected_idx.size > 0:
            y_train[selected_idx] = 0
            y_train[selected_idx, c] = 1
            r_training_idx_list.extend(selected_idx.tolist())
            r_training_label_list.extend([c] * selected_idx.size)
            train_index.extend(selected_idx.tolist())

        correct_label_count(selected_idx, c)

    print()
    train_mask = sample_mask(
        np.array(sorted(set(train_index)), dtype=np.int64), y_train.shape[0]
    )
    return (
        np.array(r_training_idx_list, dtype=np.int64),
        np.array(r_training_label_list, dtype=np.int64),
        y_train.copy(),
        train_mask.copy(),
    )




def compute_t_per_class(adj, y_train, dataset, tau=2, factor=2, r_training_cfg=None):
    r_training_cfg = _resolve_r_training_cfg(r_training_cfg)
    tau = float(r_training_cfg["budget_tau"])
    factor = float(r_training_cfg["budget_factor"])
    n = adj.shape[0]
    avg_degree = adj.sum() / n
    eta = n / (avg_degree ** tau)
    labels_per_class = y_train.sum(axis=0)
    total_labels = labels_per_class.sum()
    t = (labels_per_class * factor * eta / (total_labels + 1e-8)).astype(np.int64)
    t[t < 1] = 1
    print(f"t per class for {dataset}: {t}")
    return t


    
def run_r_training_label_expansion(adj, labels, labeled_idx, dataset="", r_training_cfg=None):
    r_training_cfg = _resolve_r_training_cfg(r_training_cfg)
    n = labels.shape[0]
    n_classes = len(np.unique(labels))

    global all_labels
    all_labels = None


    y_train = np.zeros((n, n_classes))
    for idx in labeled_idx:
        y_train[idx, labels[idx]] = 1

    train_mask = sample_mask(labeled_idx, n)
    t = compute_t_per_class(adj, y_train, dataset, r_training_cfg=r_training_cfg)

    r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask = r_training_label_expansion(
        W=adj,
        t=t,
        y_train=y_train,
        train_mask=train_mask,
        r_training_cfg=r_training_cfg,
    )
    return r_training_idx, r_training_pred, expanded_label_matrix, expanded_train_mask
