"""
vs_correlation_all.py  –  SAFE VERSION
----------------------------------------------------------
Evaluates several Vs correlations on SCPTU data and prints
cross-validated R² & RMSE (5-fold).

Just run:
    python vs_correlation_all.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────── user paths ────────────────────────────────────
ROOT_DIR  = r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\bavois\cpt_data_res"
FILE_GLOB = "SCPTU*_interpreted.csv"

FLAVOURS = ["simple", "robertson", "kruiver", "andrus", "piecewise"]

# constants
PA      = 100.0        # kPa (atmospheric)
G       = 9.81         # m/s²
EPS     = 1e-6         # tiny positive number to avoid log(0)

# basic sanity limits
MIN_RHO   = 1000       # kg/m³
MIN_SIGMA = 1          # kPa
MIN_QNET  = 1          # kPa


# ─────────────── data loader ───────────────────────────────────
def load_scptu_rows() -> pd.DataFrame:
    rows = []
    for csv in Path(ROOT_DIR).glob(FILE_GLOB):
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()

        z_vs   = pd.to_numeric(df["Z from SCPTu [m/s]"], errors="coerce")
        vs_mea = pd.to_numeric(df["Vs from SCPTu [m/s]"], errors="coerce")
        ok = (~z_vs.isna()) & (~vs_mea.isna())
        if not ok.any():
            continue

        # continuous profiles – sort by depth
        depth = pd.to_numeric(df["Depth (sbb) [m]"], errors="coerce").values
        order = np.argsort(depth)
        depth = depth[order]

        def profile(name):
            col = pd.to_numeric(df.get(name), errors="coerce").values
            return col[order]

        rho        = profile("rho (Lengkeek 2022) [kg/m3]")
        sig_eff    = profile("sigma_v_prime [kPa]")
        sig_tot    = profile("sigma_v_total (Lengkeek 2022) [kPa]")
        qc         = profile("qc (sbb) [kPa]")
        qt         = profile("qt [kPa]")
        fr         = profile("Fr [%]") / 100.0
        Ic         = profile("IC")

        for z, vs in zip(z_vs[ok], vs_mea[ok]):
            interp = lambda arr: np.interp(z, depth, arr) if arr.size else np.nan
            rows.append(dict(
                depth=z, vs=vs,
                rho=interp(rho), sigma=interp(sig_eff), sigma_tot=interp(sig_tot),
                qc=interp(qc), qt=interp(qt), fr=interp(fr), Ic=interp(Ic)
            ))

    d = pd.DataFrame(rows).dropna(subset=["vs","rho","sigma"])
    d = d[(d["rho"] > MIN_RHO) & (d["sigma"] > MIN_SIGMA)]
    d.reset_index(drop=True, inplace=True)
    return d


# ─────────────── design-matrix builders ────────────────────────
def _clip_pos(arr, lower=EPS):
    """Return array clipped to >0 and mask of rows still finite."""
    arr_c = np.clip(arr, lower, None)
    return arr_c, arr_c > lower

def build_X_y(df, flavour):
    """Return X, y as NumPy OR dict for piecewise."""
    y = np.log(df["vs"].values)

    if flavour == "simple":
        df2 = df[df["qc"] > 0].copy()
        df2["gamma"] = df2["rho"] * G / 1000      # kN/m³
        gam, m1 = _clip_pos(df2["gamma"].values)
        sig, m2 = _clip_pos(df2["sigma"].values)
        qc , m3 = _clip_pos(df2["qc"].values)
        mask = m1 & m2 & m3
        X = np.log(np.column_stack([gam[mask], sig[mask], qc[mask]]))
        return X, np.log(df2["vs"].values[mask])

    if flavour == "robertson":
        qnet = (df["qt"] - df["sigma_tot"]).clip(lower=MIN_QNET)
        Ic   = df["Ic"].fillna(2.6)
        qnet, m1 = _clip_pos(qnet.values)
        Ic_v, m2 = _clip_pos(Ic.values)
        mask = m1 & m2
        X = np.column_stack([Ic_v[mask], np.log(qnet[mask] / PA)])
        return X, y[mask]

    if flavour == "kruiver":
        qnet = ((df["qt"] - df["sigma_tot"]).clip(lower=MIN_QNET))/1000  # MPa
        fr   = df["fr"].clip(lower=EPS)
        sigM = df["sigma"]/1000
        qnet, m1 = _clip_pos(qnet.values)
        fr  , m2 = _clip_pos(fr.values)
        sigM, m3 = _clip_pos(sigM.values)
        mask = m1 & m2 & m3
        X = np.log(np.column_stack([qnet[mask], fr[mask], sigM[mask]]))
        return X, y[mask]

    if flavour == "andrus":
        df2 = df.dropna(subset=["qt","Ic"]).copy()
        qt , m1 = _clip_pos(df2["qt"].values)
        Ic , m2 = _clip_pos(df2["Ic"].values)
        dep, m3 = _clip_pos(df2["depth"].values)
        mask = m1 & m2 & m3
        X = np.log(np.column_stack([qt[mask], Ic[mask], dep[mask]]))
        return X, np.log(df2["vs"].values[mask])

    if flavour == "piecewise":
        df2 = df.copy()
        df2["gamma"] = df2["rho"] * G / 1000
        df2["group"] = np.where(df2["Ic"] < 2.6, "sand", "clay")
        out = {}
        for grp in ("sand", "clay"):
            sub = df2[(df2["group"] == grp) & (df2["qc"] > 0)]
            gam, m1 = _clip_pos(sub["gamma"].values)
            sig, m2 = _clip_pos(sub["sigma"].values)
            qc , m3 = _clip_pos(sub["qc"].values)
            mask = m1 & m2 & m3
            if mask.any():
                Xg = np.log(np.column_stack([gam[mask], sig[mask], qc[mask]]))
                yg = np.log(sub["vs"].values[mask])
                out[grp] = (Xg, yg)
        return out

    raise ValueError(flavour)


# ─────────────── CV helper ──────────────────────────────────────
def cv_r2_rmse(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2s, rmses = [], []
    for train, test in kf.split(X):
        mdl = LinearRegression().fit(X[train], y[train])
        y_hat = mdl.predict(X[test])
        r2s.append(r2_score(np.exp(y[test]), np.exp(y_hat)))
        rmses.append(np.sqrt(mean_squared_error(np.exp(y[test]), np.exp(y_hat))))
    return np.mean(r2s), np.mean(rmses)


# ─────────────── main run ───────────────────────────────────────
if __name__ == "__main__":
    df_all = load_scptu_rows()
    if df_all.empty:
        raise RuntimeError("No Vs observations after cleaning.")

    results = []
    for flav in FLAVOURS:
        if flav == "piecewise":
            piece = build_X_y(df_all, flav)
            r2w = rmw = n_sum = 0
            for grp, (Xg, yg) in piece.items():
                r2, rm = cv_r2_rmse(Xg, yg)
                n      = len(yg)
                r2w   += r2 * n
                rmw   += rm * n
                n_sum += n
            results.append([flav, n_sum, r2w/n_sum, rmw/n_sum])
        else:
            X, y = build_X_y(df_all, flav)
            r2, rm = cv_r2_rmse(X, y)
            results.append([flav, len(y), r2, rm])

    results.sort(key=lambda r: r[3])  # lowest RMSE first

    print("\nCross-validated performance (5-fold)")
    print("Model        n    R²    RMSE [m/s]")
    for m, n, r2, rm in results:
        print(f"{m:<10} {n:5d}  {r2:0.3f}   {rm:8.1f}")
