from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def profile(df, name):
    col = pd.to_numeric(df.get(name), errors="coerce").values
    depth = pd.to_numeric(df["Depth (sbb) [m]"], errors="coerce").values
    order = np.argsort(depth)
    return depth[order], col[order]

def kruiver_vs_manual(qt, friction, effective_stress, total_stress, rho,
                      a=359.0, b=0.119, c=0.100, d=0.204, eps=1e-6):
    qnet = np.clip(qt - total_stress, eps, None) / 1000  # MPa
    fr   = np.clip(friction, eps, None) / 1000           # MPa
    sig  = np.clip(effective_stress, eps, None) / 1000   # MPa
    vs = a * (qnet ** b) * (fr ** c) * (sig ** d)
    G0 = rho * vs**2
    return vs, G0


def plot_vs_profiles(depth, vs_profiles, z_measured, vs_measured, vs_custom=None, label_custom="Modified Kruiver"):
    plt.figure(figsize=(6, 8))
    for label, vs in vs_profiles.items():
        if vs is not None and np.any(~np.isnan(vs)):
            plt.plot(vs, depth, label=label)
    if vs_custom is not None:
        plt.plot(vs_custom, depth, '--', label=label_custom, linewidth=2)
    plt.scatter(vs_measured, z_measured, color='black', s=30, label="Measured Vs", zorder=5)
    plt.gca().invert_yaxis()
    plt.xlabel("Vs [m/s]")
    plt.ylabel("Depth [m]")
    plt.grid(True)
    plt.legend()
    plt.title("Vs Profiles Comparison")
    plt.tight_layout()
    plt.show()

# === MAIN ===
ROOT_DIR = r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\bavois\cpt_data_res"
FILE_GLOB = "SCPTU*_interpreted.csv"

# 🔧 CPT TO LOAD – change this to test another one
CPT_ID = "SCPTU03"

# Find matching file
matching_files = [f for f in Path(ROOT_DIR).glob(FILE_GLOB) if CPT_ID in f.name]
if not matching_files:
    raise FileNotFoundError(f"No file found for CPT ID: {CPT_ID}")
csv_path = matching_files[0]

print(f"Processing: {csv_path.name}")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Measured Vs
z_vs   = pd.to_numeric(df["Z from SCPTu [m]"], errors="coerce")
vs_mea = pd.to_numeric(df["Vs from SCPTu [m/s]"], errors="coerce")

# Continuous profiles
depth, rho     = profile(df, "rho (Lengkeek 2022) [kg/m3]")
_, qt          = profile(df, "qt [kPa]")
_, fs          = profile(df, "fs (sbb) [kPa]")  # sleeve friction
_, total_stress = profile(df, "sigma_v_total (Lengkeek 2022) [kPa]")
_, eff_stress   = profile(df, "sigma_v_prime [kPa]")

vs_profiles = {
    #"Robertson (2014)": profile(df, "Vs (Robertson and Cabal 2014) [m/s]")[1],
    #"Mayne (2007)":     profile(df, "Vs (Mayne 2007) [m/s]")[1],
    #"Zhang (2017)":     profile(df, "Vs (Zhang and Tong 2017) [m/s]")[1],
    #"Ahmed (2017)":     profile(df, "Vs (Ahmed 2017) [m/s]")[1],
    "Kruiver (2020)":   profile(df, "Vs (Kruiver et al 2020) [m/s]")[1],
}

# Custom correlation
vs_custom, _ = kruiver_vs_manual(
    qt=qt,
    friction=fs,
    effective_stress=eff_stress,
    total_stress=total_stress,
    rho=rho,
    #a=359.0, b=0.119, c=0.100, d=0.204,
    a=250, b=0.2, c=0.0, d=0.204,
)

# Plot
plot_vs_profiles(depth, vs_profiles, z_vs, vs_mea, vs_custom)
