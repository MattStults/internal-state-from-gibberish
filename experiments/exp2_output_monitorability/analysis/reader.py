"""exp2: the reader that turns features-at-each-budget into held-out predicted concept distributions.

reader_proba_by_budget fits a FIXED-CAPACITY K-way classifier (StandardScaler -> PCA -> balanced multinomial
logistic; caps pinned so the recovered bits aren't inflated by capacity) and returns, per budget, the
STREAM-LEVEL held-out (cross-validated) predicted concept distributions. Feeds info.bits_recovered / the rate
estimator. Reader-agnostic to HOW features were built (sampled-token chars, or distribution-access gen_topk) --
that's a separate unit; this is the currency-consistent fit + CV glue.
"""
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _channel_pipeline(kind, n_features, n_train_eff, seed=0):
    """A pipeline matched to the channel's feature TYPE, with a capacity grid. dense: continuous distribution
    features (StandardScaler->PCA->Logistic; PCA dims IN the grid). sparse: one-hot/count token features --
    NO StandardScaler (dividing sparse counts by tiny stds overfits); TF-IDF (sublinear) then Logistic.
    n_train_eff = the effective inner-CV TRAIN size, so PCA n_components can't exceed it (avoids a crash).
    seed: pins PCA's randomized-SVD draw. At the real run shape (288 samples, d_model features) sklearn's auto
    solver picks randomized SVD, so an unseeded PCA jitters the recovered bits ~3e-3 run-to-run (material at
    3 dp, and it rides the R_emb reader = the headline dist-R_emb gap). random_state=seed makes it reproducible."""
    if kind == "dense":
        cap = max(2, min(n_features, n_train_eff - 1))
        dims = sorted({d for d in (20, 30) if d <= cap}) or [cap]
        pipe = make_pipeline(StandardScaler(), PCA(random_state=seed), LogisticRegression(max_iter=2000, class_weight="balanced"))
        # C down to 1e-3 so a weak/thin-data signal picks strong regularization and FLOORS at ~0 held-out bits
        # (predicts the prior) rather than overfitting into an invalid NEGATIVE lower bound.
        grid = {"pca__n_components": dims, "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    elif kind == "sparse":
        pipe = make_pipeline(TfidfTransformer(sublinear_tf=True), LogisticRegression(max_iter=2000, class_weight="balanced"))
        grid = {"logisticregression__C": [0.03, 0.3, 3.0]}
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return pipe, grid


def best_reader_proba_by_budget(features_by_budget, y_true, budgets, kind="dense", folds=5, inner_folds=3, seed=0,
                                n_jobs=-1):
    """Like reader_proba_by_budget, but a BEST-DECODER-for-its-channel: uses a channel-appropriate pipeline
    (kind='dense'|'sparse') and selects capacity (C) by INNER CV that maximizes held-out bits (neg_log_loss),
    reporting OUTER-CV held-out proba (nested CV -- capacity is not tuned on the reported score). So each
    reader is a valid held-out lower bound for its OWN channel; only then is a dist-sampled gap meaningful.
    n_jobs: inner-CV parallelism; set to 1 when the CALLER parallelizes across (reader, budget) tasks, so the
    outer fan-out saturates all cores instead of nested n_jobs=-1 oversubscribing to just the innermost level."""
    y = np.asarray(y_true)
    if kind not in ("dense", "sparse"):
        raise ValueError(f"unknown kind {kind!r}")
    if not np.array_equal(np.unique(y), np.arange(y.max() + 1)):
        raise ValueError("y_true must be contiguous labels 0..K-1")
    if np.bincount(y).min() < folds:
        raise ValueError(f"every class needs >= folds ({folds}) streams for stream-level CV")
    n = len(y)
    n_train_eff = int(n * (folds - 1) / folds * (inner_folds - 1) / inner_folds)   # PCA fits on this many
    out = {}
    for T in budgets:
        X = np.asarray(features_by_budget[T], dtype=float)
        pipe, grid = _channel_pipeline(kind, X.shape[1], n_train_eff, seed=seed)
        gs = GridSearchCV(pipe, grid, scoring="neg_log_loss", n_jobs=n_jobs,
                          cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed))
        outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
        out[T] = cross_val_predict(gs, X, y, cv=outer, method="predict_proba", n_jobs=n_jobs)
    return out


def reader_proba_by_budget(features_by_budget, y_true, budgets, pca_dims=30, C=1.0, folds=5, seed=0):
    """features_by_budget[T]: (n_streams, d) features at budget T. y_true: (n_streams,) concept labels in
    [0, K) (columns of the returned proba are class 0..K-1). Returns {T: (n_streams, K) STREAM-LEVEL held-out
    (cross-validated) predicted concept distributions} from a fixed-capacity reader (StandardScaler -> PCA(dims)
    -> balanced multinomial logistic; pca_dims/C/folds pinned). Cross-validation is what makes bits_recovered a
    valid held-out lower bound rather than a memorized in-sample over-estimate.

    Preconditions (enforced): y_true is contiguous labels 0..K-1 (remap bits-ladder groupings BEFORE calling,
    else column order won't match); every class has >= folds streams. NOT enforced: common-N across classes/
    scales -- subsample per concept in the caller/runner, since more streams => tighter decoder => more
    recovered bits (class_weight='balanced' handles the loss, not the training QUANTITY)."""
    y = np.asarray(y_true)
    if not np.array_equal(np.unique(y), np.arange(y.max() + 1)):
        raise ValueError("y_true must be contiguous labels 0..K-1 (remap groupings before calling)")
    counts = np.bincount(y)
    if counts.min() < folds:
        raise ValueError(f"every class needs >= folds ({folds}) streams for stream-level CV; "
                         f"min class count is {counts.min()}")
    out = {}
    for T in budgets:
        X = np.asarray(features_by_budget[T], dtype=float)
        ndim = int(min(pca_dims, X.shape[1], X.shape[0] - 1))
        if ndim < pca_dims:
            warnings.warn(f"PCA capacity dropped to {ndim} < pca_dims={pca_dims} -- capacity not pinned "
                          "(features/samples too few)")
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=ndim, random_state=seed),
            LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        out[T] = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    return out
