import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import skew
import numpy as np

def collect_statistics_summary(layer_df: pd.DataFrame, layer_name: str, cpt_name: str, site_name: str) -> list[dict]:
    variables = [
        "rho (Lengkeek 2022) [kg/m3]",
        "Poisson ratio gwl [-]",
        "E0 (Ahmed 2017) [kPa]",
        "E0 (Kruiver et al 2020) [kPa]",
        "E0 (Robertson and Cabal 2014) [kPa]",

    ]

    summaries = []

    for var in variables:
        if var not in layer_df.columns:
            continue

        data = layer_df[var].dropna()
        if data.empty:
            continue

        n = len(data)
        mean = data.mean()
        median = data.median()
        std = data.std()
        min_val = data.min()
        max_val = data.max()
        skew_val = skew(data)
        cv = std / mean if mean != 0 else np.nan

        # Outlier detection
        outliers_std = data[(data > mean + 3*std) | (data < mean - 3*std)]
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        outliers_iqr = data[(data > data.quantile(0.75) + 1.5 * iqr) | (data < data.quantile(0.25) - 1.5 * iqr)]

        # Trust flag logic
        flags = []
        if n < 5:
            flags.append("Unreliable")
        if abs(mean - median) > 0.3 * std:
            flags.append("Skewed")
        if std > 0.5 * mean:
            flags.append("Noisy")
        if abs(skew_val) > 1:
            flags.append("High Skew")
        if cv > 0.5:
            flags.append("High CV")
        if len(outliers_std) > 3 or len(outliers_iqr) > 3:
            flags.append("Outliers")
        if not flags:
            flags.append("OK")

        summaries.append({
            "Site": site_name,
            "CPT": cpt_name,
            "Layer": layer_name,
            "Variable": var,
            "n": n,
            "Mean": round(mean, 2),
            "Median": round(median, 2),
            "Std": round(std, 2),
            "Min": round(min_val, 2),
            "Max": round(max_val, 2),
            "Skewness": round(skew_val, 2),
            "CV": round(cv, 2),
            "Outliers_std>3": len(outliers_std),
            "Outliers_IQR": len(outliers_iqr),
            "Trust_Flag": ", ".join(flags)
        })

    return summaries



def get_layer_bounds(layer_string: str, max_depth: float) -> list[tuple[float, float]]:
    """
    Convert a comma-separated string of depths into layer boundaries.

    Args:
        layer_string (str): Comma-separated string of horizontal layer depths (e.g., "0, 0.6, 3.4").
        max_depth (float): Maximum depth of the CPT (used to close the last layer).

    Returns:
        list[tuple[float, float]]: List of (top, bottom) depth intervals for each layer.
    """
    values = [float(v.strip()) for v in layer_string.split(",")]
    if values[-1] < max_depth:
        values.append(max_depth)
    return [(values[i], values[i + 1]) for i in range(len(values) - 1)]


def extract_cpt_id_parts(filename: str) -> tuple[str, str]:
    """
    Extract the CPT name and site name from a CPT result CSV filename.

    Args:
        filename (str): Filename like 'CPTU01_bavois_interpreted.csv'.

    Returns:
        tuple[str, str]: Tuple of (cpt_name, site).
    """
    name, site = os.path.basename(filename).replace(".csv", "").split("_")[:2]
    return name, site


def check_statistics(layer_df: pd.DataFrame, layer_name: str, cpt_name: str, site_name: str, save_dir: str):
    """
    Perform basic statistical checks and generate boxplots for selected variables.

    Args:
        layer_df (pd.DataFrame): Subset of data for a single layer.
        layer_name (str): e.g., 'layer1'
        cpt_name (str): CPT name (e.g., 'CPTU01')
        site_name (str): Site name (e.g., 'bavois')
        save_dir (str): Where to save plots.
    """
    variables = [
        "rho (Lengkeek 2022) [kg/m3]",
        "G0 (Ahmed 2017) [MPa]",
        "Poisson ratio gwl [-]"
    ]

    fig, axs = plt.subplots(len(variables), 1, figsize=(8, 2.5 * len(variables)))
    if len(variables) == 1:
        axs = [axs]

    print(f"\n[QC] {site_name} | {cpt_name} | {layer_name} | n={len(layer_df)}")

    for ax, var in zip(axs, variables):
        if var not in layer_df.columns:
            print(f"  [SKIP] Missing: {var}")
            continue

        data = layer_df[var].dropna()
        if data.empty:
            print(f"  [SKIP] Empty: {var}")
            continue

        mean = data.mean()
        median = data.median()
        std = data.std()
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        outliers_std = data[(data > mean + 3 * std) | (data < mean - 3 * std)]
        outliers_iqr = data[(data > data.quantile(0.75) + 1.5 * iqr) | (data < data.quantile(0.25) - 1.5 * iqr)]

        print(f"  {var}")
        print(f"    Mean = {mean:.2f}, Median = {median:.2f}, Std = {std:.2f}")
        print(f"    Min = {data.min():.2f}, Max = {data.max():.2f}")
        print(f"    Outliers (std) = {len(outliers_std)}, Outliers (IQR) = {len(outliers_iqr)}")

        ax.boxplot(data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
        ax.set_title(var)
        ax.set_xlabel(var)

    fig.suptitle(f"{site_name} - {cpt_name} - {layer_name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{site_name}_{cpt_name}_{layer_name}_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved boxplot → {out_path}")


def compute_layer_stats(df: pd.DataFrame, top: float, bottom: float, method: str = "mean",
                        correlation: str = 'Ahmed 2017') -> dict:
    """
    Compute layer-wise statistics between two depths using a specified aggregation method.

    Args:
        df (pd.DataFrame): DataFrame containing CPT results.
        top (float): Upper depth of the layer.
        bottom (float): Lower depth of the layer.
        method (str, optional): Aggregation method. One of 'mean', 'median', or 'mode'. Defaults to 'mean'.
        correlation (str, optional): Correlation method for G0 and E0. Defaults to 'Ahmed 2017'.

    Returns:
        dict: Dictionary with layer thickness, mean/std of density, G0, and Poisson ratio.
    """
    mask = (df["Depth (sbb) [m]"] >= top) & (df["Depth (sbb) [m]"] < bottom)
    layer_df = df[mask]
    thickness = bottom - top

    if method == "mean":
        agg = layer_df.mean(numeric_only=True)
    elif method == "median":
        agg = layer_df.median(numeric_only=True)
    elif method == "mode":
        mode_df = layer_df.mode(numeric_only=True)
        agg = mode_df.iloc[0] if not mode_df.empty else layer_df.mean(numeric_only=True)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Ensure the correlation column exists and its from the correct options
    if f"G0 ({correlation}) [kPa]" not in layer_df.columns or f"E0 ({correlation}) [kPa]" not in layer_df.columns:
        raise ValueError(f"Correlation method '{correlation}' not found in layer data.")
    if correlation not in ['Robertson and Cabal 2014', 'Mayne 2007', 'Zhang and Tong 2017', 'Ahmed 2017',
                           'Kruiver et al 2020']:
        raise ValueError(
            f"Unsupported correlation method: {correlation}, choose between 'Robertson and Cabal 2014', 'Mayne 2007', 'Zhang and Tong 2017', 'Ahmed 2017', 'Kruiver et al 2020'.")

    return {
        "Thickness (m)": thickness,
        "Thickness_std (m)": 0.0,
        "Density (kg/m3)": agg["rho (Lengkeek 2022) [kg/m3]"],
        "Density_std (kg/m3)": layer_df["rho (Lengkeek 2022) [kg/m3]"].std(),
        "G0 (kPa)": agg[f"G0 ({correlation}) [kPa]"],
        "G0_std (kPa)": layer_df[f"G0 ({correlation}) [kPa]"].std(),
        "Poisson (-)": agg["Poisson ratio gwl [-]"],
        "Poisson_std (-)": layer_df["Poisson ratio gwl [-]"].std(),
        "E0 (kPa)": agg[f"E0 ({correlation}) [kPa]"],
        "E0_std (kPa)": layer_df[f"E0 ({correlation}) [kPa]"].std(),
    }


def process_cpt_file(csv_path: str, layering_df: pd.DataFrame, method: str = "mean", correlation: str = "Ahmed 2017") -> tuple[list[dict], list[dict]]:
    df = pd.read_csv(csv_path)
    cpt_name, site = extract_cpt_id_parts(csv_path)
    max_depth = df["Depth (sbb) [m]"].max()

    match = layering_df[
        (layering_df["cpt_name"].str.lower() == cpt_name.lower()) &
        (layering_df["site"].str.lower() == site.lower())
    ]
    if match.empty:
        print(f"[WARNING] No layering data for {cpt_name} at {site}")
        return [], []

    layer_bounds = get_layer_bounds(match.iloc[0]["horiz_lines"], max_depth)
    results = []
    qc_summaries = []

    for i, (top, bottom) in enumerate(layer_bounds):
        layer_df = df[(df["Depth (sbb) [m]"] >= top) & (df["Depth (sbb) [m]"] < bottom)]
        layer_name = f"layer{i + 1}"

        # Add QC summary (skewness, CV, flags)
        qc_summary_rows = collect_statistics_summary(layer_df, layer_name, cpt_name, site)
        qc_summaries.extend(qc_summary_rows)

        # Compute layer statistics (mean, std, etc.)
        stats = compute_layer_stats(df, top, bottom, method=method, correlation=correlation)
        results.append({
            "Name": cpt_name,
            "Layer": layer_name,
            **stats
        })

    return results, qc_summaries


# === USER CONFIG ===
layering_csv = r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\layering.csv"
site_folders = [
    r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\bavois\cpt_data_res",
    r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\chavornay\cpt_data_res",
    r"N:\Projects\11211500\11211717\B. Measurements and calculations\cpt_results\ependes\cpt_data_res"
]

aggregation_method = "mean"  # Can also be 'median' or 'mode'
correlation_methods = ["Ahmed 2017", "Kruiver et al 2020", "Robertson and Cabal 2014"]

# === LOAD LAYERING CSV ===
layering_df = pd.read_csv(layering_csv)


import re
# === PROCESS EACH FOLDER ===
for folder in site_folders:
    site_name = os.path.basename(folder).lower()
    csv_files = glob.glob(os.path.join(folder, "*.csv"))

    # Only compute QC summary once (doesn't depend on correlation)
    all_qc = []

    for csv_path in csv_files:
        if "interpreted" not in os.path.basename(csv_path).lower():
            continue

        _, qc_summary = process_cpt_file(csv_path, layering_df, method=aggregation_method,
                                         correlation=correlation_methods[0])
        all_qc.extend(qc_summary)

    # Save the single QC summary
    if all_qc:
        qc_df = pd.DataFrame(all_qc)
        qc_output = os.path.join(folder, f"{site_name}_qc_summary.csv")
        qc_df.to_csv(qc_output, sep=";", index=False)
        print(f"[INFO] Saved QC summary → {qc_output}")

    # Now run the correlation loop only for actual E0/G0 stats
    for correlation in correlation_methods:
        all_results = []

        for csv_path in csv_files:
            if "interpreted" not in os.path.basename(csv_path).lower():
                continue

            cpt_results, _ = process_cpt_file(csv_path, layering_df, method=aggregation_method, correlation=correlation)
            all_results.extend(cpt_results)

        corr_short = re.sub(r'[^a-z0-9_]', '', correlation.lower().replace(" ", "_"))

        if all_results:
            output_df = pd.DataFrame(all_results)
            output_path = os.path.join(folder, f"{site_name}_cpt_statistics_{corr_short}.csv")
            output_df.to_csv(output_path, sep=";", index=False)
            print(f"[INFO] Saved → {output_path}")


        else:
            print(f"[WARNING] No results for site '{site_name}' using correlation '{correlation}'")
