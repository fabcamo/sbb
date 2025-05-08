import os
import glob
import pandas as pd

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


def compute_layer_stats(df: pd.DataFrame, top: float, bottom: float, method: str = "mean") -> dict:
    """
    Compute layer-wise statistics between two depths using a specified aggregation method.

    Args:
        df (pd.DataFrame): DataFrame containing CPT results.
        top (float): Upper depth of the layer.
        bottom (float): Lower depth of the layer.
        method (str, optional): Aggregation method. One of 'mean', 'median', or 'mode'. Defaults to 'mean'.

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

    return {
        "Thickness": thickness,
        "Density": agg["rho (Lengkeek 2022) [kg/m3]"],
        "G0": agg["G0 (Ahmed 2017) [MPa]"],
        "Poisson": agg["Poisson ratio gwl [-]"],
        "Thickness_std": 0.0,
        "Density_std": layer_df["rho (Lengkeek 2022) [kg/m3]"].std(),
        "G0_std": layer_df["G0 (Ahmed 2017) [MPa]"].std(),
        "Poisson_std": layer_df["Poisson ratio gwl [-]"].std(),
    }


def process_cpt_file(csv_path: str, layering_df: pd.DataFrame, method: str = "mean") -> list[dict]:
    """
    Process a single CPT CSV file and compute statistics for each defined layer.

    Args:
        csv_path (str): Path to the CPT interpreted CSV file.
        layering_df (pd.DataFrame): DataFrame containing site, cpt_name, and layer boundaries.
        method (str, optional): Aggregation method to use ('mean', 'median', 'mode'). Defaults to 'mean'.

    Returns:
        list[dict]: List of dictionaries with computed statistics for each layer.
    """
    df = pd.read_csv(csv_path)
    cpt_name, site = extract_cpt_id_parts(csv_path)
    max_depth = df["Depth (sbb) [m]"].max()

    match = layering_df[
        (layering_df["cpt_name"].str.lower() == cpt_name.lower()) &
        (layering_df["site"].str.lower() == site.lower())
    ]
    if match.empty:
        print(f"[WARNING] No layering data for {cpt_name} at {site}")
        return []

    layer_bounds = get_layer_bounds(match.iloc[0]["horiz_lines"], max_depth)
    results = []
    for i, (top, bottom) in enumerate(layer_bounds):
        stats = compute_layer_stats(df, top, bottom, method=method)
        results.append({
            "Name": cpt_name,
            "Layer": f"layer{i+1}",
            **stats
        })
    return results



# === USER CONFIG ===
layering_csv = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\layering.csv"
site_folders = [
    r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois",
    r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\chavornay",
    r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\ependes"
]
aggregation_method = "mean"  # Can also be 'median' or 'mode'

# === LOAD LAYERING CSV ===
layering_df = pd.read_csv(layering_csv)

# === PROCESS EACH FOLDER ===
for folder in site_folders:
    site_name = os.path.basename(folder).lower()
    csv_files = glob.glob(os.path.join(folder, "*.csv"))

    all_results = []
    for csv_path in csv_files:
        if "interpreted" not in os.path.basename(csv_path).lower():
            continue  # Skip non-result files
        cpt_results = process_cpt_file(csv_path, layering_df, method=aggregation_method)
        all_results.extend(cpt_results)

    if all_results:
        output_df = pd.DataFrame(all_results)
        output_path = os.path.join(folder, f"{site_name}_cpt_statistics.csv")
        output_df.to_csv(output_path, sep=";", index=False)
        print(f"[INFO] Saved → {output_path}")
    else:
        print(f"[WARNING] No results for site: {site_name}")
