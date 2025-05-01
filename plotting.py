import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np


def load_all_csvs(csv_folder_path):
    """
    Load all CSV files in a folder into a dictionary of DataFrames.

    Args:
        csv_folder_path (str): Path to folder containing CSV files.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping file stem (no extension) to its DataFrame.
    """
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))
    if not csv_files:
        print("[ERROR] No CSV files found.")
        return {}

    data_dict = {}
    for csv_file in csv_files:
        try:
            key = os.path.basename(csv_file).replace('.csv', '')
            data_dict[key] = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_file}: {e}")

    print(f"[INFO] Loaded {len(data_dict)} CSVs.")
    return data_dict


def plot_multi_param_vs_depth(data_dict, parameters, save_folder, label_dict=None):
    """
    Plot any number of parameters vs depth for each CPT file in one figure.
    Each subplot uses the first available parameter as its x-axis label
    and shows horizontal gridlines every 0.5 m.

    Args:
        data_dict (dict[str, pd.DataFrame]): Dictionary mapping CPT ID to DataFrame.
        parameters (list[str]): List of column names to plot against depth.
        save_folder (str): Path where the final figure will be saved.
        label_dict (dict[str, str], optional): Mapping of column names to display labels.
    """
    if not data_dict:
        print("[ERROR] No data to plot.")
        return

    n_cpts = len(data_dict)
    fig, axs = plt.subplots(1, n_cpts, figsize=(4 * n_cpts, 10))

    if n_cpts == 1:
        axs = [axs]

    line_styles = ['-', '--', ':', '-.']
    colors = plt.cm.tab10.colors

    for ax, (cpt_id, df) in zip(axs, data_dict.items()):
        if 'Depth (sbb) [m]' not in df.columns:
            print(f"[WARNING] 'Depth (sbb) [m]' not found in {cpt_id}, skipping.")
            continue

        depth = df['Depth (sbb) [m]']
        used_params = []

        for i, param in enumerate(parameters):
            if param not in df.columns:
                print(f"[WARNING] {param} not found in {cpt_id}, skipping this parameter.")
                continue

            values = df[param]
            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]

            label = label_dict.get(param, param) if label_dict else param
            ax.plot(values, depth, label=label, linestyle=style, linewidth=1.5, color=color)
            used_params.append(param)

        # Set custom ticks every 0.5 m
        min_depth, max_depth = depth.min(), depth.max()
        yticks = np.arange(np.floor(min_depth), np.ceil(max_depth) + 0.5, 0.5)
        ax.set_yticks(yticks)

        ax.set_title(cpt_id.replace('_interpreted', ''), fontsize=10)

        if used_params:
            display_labels = [label_dict.get(p, p) for p in used_params]
            xlabel = extract_common_label(display_labels)
            ax.set_xlabel(xlabel, fontsize=10)
        else:
            ax.set_xlabel("Parameter value", fontsize=10)

        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.grid(True, axis='y', which='major')  # Only horizontal gridlines
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="best")

    param_clean = "_".join([re.sub(r"[^\w\-]", "_", p) for p in parameters])
    plt.suptitle(f"{', '.join([label_dict.get(p, p) for p in parameters])} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{param_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")


import re

def extract_common_label(param_labels):
    """
    Extract a meaningful common label from multiple display strings.
    Preserves trailing units like [kPa], [-], etc., if shared.
    """
    if not param_labels:
        return "Parameter value"
    if len(param_labels) == 1:
        return param_labels[0]

    # Find longest common prefix
    def longest_common_prefix(strings):
        prefix = strings[0]
        for s in strings[1:]:
            i = 0
            while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
                i += 1
            prefix = prefix[:i]
        return prefix.strip()

    # Find shared trailing units like [kPa], [m/s], [-]
    def shared_unit_suffix(labels):
        units = [re.findall(r"\[[^]]+\]$", lbl) for lbl in labels]
        units = [u[0] for u in units if u]
        return units[0] if len(units) == len(labels) and all(u == units[0] for u in units) else ""

    prefix = longest_common_prefix(param_labels)
    suffix = shared_unit_suffix(param_labels)

    # Clean middle
    prefix = prefix.rstrip(" (-/")

    # Final label
    return f"{prefix} {suffix}".strip()
