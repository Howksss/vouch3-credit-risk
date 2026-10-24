"""Export the modelling table to dataset/semiannual.csv.gz, INN replaced by a keyed hash."""
import hashlib
import hmac
import os
import secrets

import model as M

KEEP = ["period", "observable", "is_spv", "is_fin", "is_cb", "default_prior"]


def main():
    if not os.path.exists(os.path.join(M.BASE, "features.csv")):
        raise SystemExit("data/features.csv not found — run the pipeline first")
    M.STRESS = M.STRESS_ALL
    df = M.load()
    M.STRESS = {"default"}
    y_default = M.load()["y"].values

    key = secrets.token_bytes(32)
    out = df[["inn", "as_of"] + M.FIN + M.DISC + M.MKT + KEEP].copy()
    out["y_stress"] = df["y"].values
    out["y_default"] = y_default
    out.insert(0, "issuer_id", [hmac.new(key, inn.encode(), hashlib.sha256).hexdigest()[:16]
                                for inn in out.pop("inn")])

    os.makedirs(os.path.dirname(M.DATASET), exist_ok=True)
    out.to_csv(M.DATASET, index=False, compression="gzip")
    print(f"saved {M.DATASET}: rows={len(out)} issuers={out['issuer_id'].nunique()} "
          f"cols={out.shape[1]}")


if __name__ == "__main__":
    main()
