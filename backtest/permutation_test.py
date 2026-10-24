"""Guard against a hidden leak: shuffle the labels and re-run the same pipeline.

If the machinery is clean, AUC on shuffled labels collapses to ~0.5. Anything
consistently above that would mean the split, the features, or the universe filter
is carrying information it should not.
"""
import numpy as np

import model as M


def run(seed=None):
    df = M.load()
    u = M.universe(df)
    train, test = M.split(u)
    train, test = train.copy(), test.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        # shuffle labels within each fold, preserving the event count
        train["y"] = rng.permutation(train["y"].values)
        test["y"] = rng.permutation(test["y"].values)
    feats = M.FIN + M.DISC
    Xtr, Xte, _ = M.prep(train, test, feats)
    w = M.fit_logit(Xtr, train["y"].values.astype(float))
    return M.auc(test["y"].values, Xte @ w)


real = run(None)
print(f"real labels        AUC = {real:.3f}")
shuffled = [run(s) for s in range(20)]
print(f"shuffled labels    mean = {np.mean(shuffled):.3f}  sd = {np.std(shuffled):.3f}  "
      f"max = {np.max(shuffled):.3f}")
print(f"real above shuffled in {sum(1 for s in shuffled if real > s)}/20 draws")
