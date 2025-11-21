import numpy as np
from scipy import optimize, stats
from scipy.spatial.distance import pdist, squareform

try:
    from sklearn.neighbors import NearestNeighbors
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

# =========================
#   Preprocessing helpers
# =========================

def _boxcox_per_column(X, min_positive_shift=1e-6, lambda_bounds=(-5.0, 5.0)):
    """Apply Box-Cox to each column, shifting non-positive values when needed.

    ``scipy.stats.boxcox`` can emit warnings when the unconstrained optimum for
    ``lambda`` would overflow ``float64``. We avoid the warning by estimating
    the maximum-likelihood lambda ourselves, constraining it to a reasonable
    range, and then applying the transform with the fixed lambda.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    X_bc = np.empty_like(X)
    shifts = np.zeros(k, dtype=float)
    lambdas = np.full(k, np.nan, dtype=float)
    is_constant = np.zeros(k, dtype=bool)

    for j in range(k):
        col = X[:, j].astype(float)

        if np.allclose(col, col[0]):
            X_bc[:, j] = col
            is_constant[j] = True
            continue

        min_val = np.min(col)
        shift = 0.0
        if min_val <= 0:
            shift = -(min_val) + min_positive_shift
            col = col + shift
        shifts[j] = shift

        def _neg_llf(lmb):
            ll = stats.boxcox_llf(lmb, col)
            return np.inf if not np.isfinite(ll) else -ll

        if lambda_bounds is None:
            res = optimize.minimize_scalar(_neg_llf)
            lo, hi = None, None
        else:
            lo, hi = lambda_bounds
            res = optimize.minimize_scalar(_neg_llf, bounds=(lo, hi), method="bounded")

        lam = float(res.x) if np.isfinite(res.x) else 1.0
        if not res.success:
            lam = 1.0

        if lo is not None:
            lam = float(np.clip(lam, lo, hi))

        X_bc[:, j] = stats.boxcox(col, lmbda=lam)
        lambdas[j] = lam

    meta = {"shifts": shifts, "lambdas": lambdas, "is_constant": is_constant}
    return X_bc, meta


def _zscore(X, eps=1e-12):
    """Column-wise z-score with constant columns forced to zero."""
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd_safe = np.where(sd < eps, 1.0, sd)
    Xn = (X - mu) / sd_safe
    # Zero-out strictly-constant columns exactly (nice for reproducibility)
    Xn[:, sd < eps] = 0.0
    return Xn, {"mean": mu, "std": sd}


# =======================================
#   Distance & MST (objective) helpers
# =======================================

def mst_length_dense(D_sub):
    """Prim's algorithm on a dense distance matrix."""
    n = D_sub.shape[0]
    if n <= 1:
        return 0.0

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True

    # For each node not yet in tree, track best edge to the tree.
    best = D_sub[0].copy()
    best[0] = np.inf

    total = 0.0
    for _ in range(n - 1):
        # Pick the closest node to the current tree
        j = np.argmin(best)
        w = best[j]
        total += w
        in_tree[j] = True
        best[j] = np.inf

        # Update frontier
        # Only update nodes not yet in MST
        np.minimum(best, D_sub[j], out=best)
        best[in_tree] = np.inf

    return float(total)

def pairwise_euclidean_pdist(X):
    return squareform(pdist(X, metric='euclidean'))


# ===== KNN mean-distance scorer =====

def _knn_mean_distances_sklearn(X_norm, K, n_jobs=None):
    if not _HAVE_SK:
        raise ImportError("scikit-learn not available. Install scikit-learn or use method='pdist'.")
    # n_neighbors=K+1 to include self; we'll drop self (distance 0).
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_norm)
    dists, _ = nn.kneighbors(X_norm, return_distance=True)
    return np.mean(dists[:, 1:K + 1], axis=1)


def _knn_mean_distances_pdist(X_norm, K):
    # Build full matrix (exact). Good for modest n.
    D = squareform(pdist(X_norm, metric="euclidean"))
    # Set diag to +inf so it won’t be selected among neighbors.
    np.fill_diagonal(D, np.inf)
    # Take K smallest per row efficiently
    # argpartition finds K smallest indices but we only need values; use partition on a copy
    # (avoid copying entire matrix for huge n; OK for moderate n)
    part = np.partition(D, K, axis=1)[:, :K]  # K smallest distances per row (unordered)
    return np.mean(part, axis=1)
    
# ==========================================
#   Greedy farthest-first + 1-swap improve
# ==========================================

def greedy_farthest_subset(D, N, seed_idx=None):
    """Farthest-first traversal using a precomputed distance matrix."""
    n = D.shape[0]
    if N >= n:
        return list(range(n))

    if seed_idx is None:
        # Choose the row with largest mean distance as the initial seed
        seed_idx = int(np.argmax(np.mean(D, axis=1)))

    chosen = [seed_idx]
    # Track each point's distance to the nearest chosen point
    nearest = D[:, seed_idx].copy()

    for _ in range(1, N):
        nearest[chosen] = -np.inf  # don't re-pick existing
        j = int(np.argmax(nearest))
        chosen.append(j)
        # Update nearest distances
        nearest = np.minimum(nearest, D[:, j])

    return chosen


def improve_by_one_swaps(D, chosen, max_rounds=10, max_trials_per_round=256, rng=None):
    """Hill-climb with 1-swaps to stretch the MST length."""
    rng = rng or np.random.default_rng()
    chosen = list(chosen)
    all_idx = np.arange(D.shape[0])
    not_chosen = np.setdiff1d(all_idx, chosen, assume_unique=False)

    def score(idx_list):
        sub = D[np.ix_(idx_list, idx_list)]
        return mst_length_dense(sub)

    best_score = score(chosen)

    for _ in range(max_rounds):
        improved = False
        # Randomize order to escape local structure
        rng.shuffle(not_chosen)
        rng.shuffle(chosen)

        trials = 0
        for out_idx in chosen:
            if improved or trials >= max_trials_per_round:
                break
            for in_idx in not_chosen:
                if trials >= max_trials_per_round:
                    break
                trials += 1

                candidate = chosen.copy()
                # swap out -> in
                pos = candidate.index(out_idx)
                candidate[pos] = int(in_idx)

                s = score(candidate)
                if s > best_score:
                    best_score = s
                    chosen = candidate
                    not_chosen = np.setdiff1d(all_idx, chosen, assume_unique=False)
                    improved = True
                    break

        if not improved:
            break  # local optimum reached

    return chosen

def calc_distance_matrix(X):
    X_bc, bc_meta = _boxcox_per_column(X)
    X_norm, zs_meta = _zscore(X_bc)
    return pairwise_euclidean_pdist(X_norm)

def select_diverse_rows(
    X,
    N,
    do_local_improve=True,
    improve_rounds=10,
    improve_trials_per_round=256,
    random_state=None,
    distance_matrix=None,
):
    """Pick a far-apart subset of rows with an optional local search step."""
    rng = np.random.default_rng(random_state)
    if distance_matrix is None:
        D = calc_distance_matrix(X)
    else:
        D = distance_matrix

    # 4) Greedy farthest-first
    seed_idx = int(np.argmax(np.mean(D, axis=1)))
    chosen = greedy_farthest_subset(D, N, seed_idx=seed_idx)

    # 5) Optional local improve (maximize MST length of chosen)
    if do_local_improve and N > 2:
        chosen = improve_by_one_swaps(
            D,
            chosen,
            max_rounds=improve_rounds,
            max_trials_per_round=improve_trials_per_round,
            rng=rng,
        )

    score = mst_length_dense(D[np.ix_(chosen, chosen)])

    return {"D": D, "indices": chosen, "mst_length": score}

def select_novel_rows(
    X,
    K,
    method="sklearn",
    n_jobs=None,
    return_scores=True,
):
    """Rank rows by mean distance to their K nearest neighbors."""
    if K < 1:
        raise ValueError("K must be >= 1")

    # 1) Box–Cox
    X_bc, bc_meta = _boxcox_per_column(X)
    # 2) z-score
    X_norm, zs_meta = _zscore(X_bc)

    # 3) Mean distance to KNN (excluding self)
    if method == "sklearn":
        scores = _knn_mean_distances_sklearn(X_norm, K, n_jobs=n_jobs)
    elif method == "pdist":
        scores = _knn_mean_distances_pdist(X_norm, K)
    else:
        raise ValueError("method must be 'sklearn' or 'pdist'.")

    # 4) ascending rank (nearest neighbors are *closest* on average)
    indices_sorted = np.argsort(scores)[::-1]

    out = {
        "indices_sorted": indices_sorted,
        "X_norm": X_norm,
        "boxcox_meta": bc_meta,
        "zscore_meta": zs_meta,
    }
    if return_scores:
        out["scores"] = scores
    return out
