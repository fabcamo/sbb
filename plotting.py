import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

from process_data import calculate_distance, sort_CPT_by_coordinates


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


def plot_multi_param_with_scatter(data_dict, parameters, scatter_x_col, scatter_y_col, save_folder, label_dict=None):
    """
    Variant of the multi-param plot that includes one optional scatter series (e.g. SDMT Vs).

    Args:
        data_dict (dict): Dictionary mapping CPT IDs to DataFrames.
        parameters (list): Column names to plot as lines vs depth.
        scatter_x_col (str): Name of the column for scatter x-axis values (e.g., 'Vs from SDMT').
        scatter_y_col (str): Name of the column for scatter y-axis values (e.g., 'Z from SDMT').
        save_folder (str): Directory to save the plot.
        label_dict (dict): Optional dict mapping raw column names to pretty labels.
    """
    if not data_dict:
        print("[ERROR] No data to plot.")
        return

    n_cpts = len(data_dict)
    fig, axs = plt.subplots(1, n_cpts, figsize=(5 * n_cpts, 10))
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

            label = label_dict.get(param, param) if label_dict else param
            values = df[param]
            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]

            ax.plot(values, depth, label=label, linestyle=style, linewidth=1.5, color=color)
            used_params.append(param)

        # Optional scatter
        if scatter_x_col in df.columns and scatter_y_col in df.columns:
            ax.scatter(df[scatter_x_col], df[scatter_y_col],
                       label=label_dict.get(scatter_x_col, scatter_x_col),
                       color='black', s=300, marker='|')

        # Format
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
        ax.grid(True, axis='y', which='major')
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc='best')

    param_clean = "_".join([re.sub(r"[^\w\-]", "_", p) for p in parameters + [scatter_x_col]])
    plt.suptitle(f"{', '.join([label_dict.get(p, p) for p in parameters])} + scatter vs Depth", y=1.02, fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{param_clean}_with_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")


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


def plot_lithology_columns(data_dict, lithology_column, depth_column, save_folder):
    """
    Plot lithology (discrete classes) vs depth for all CPTs in the provided dictionary.

    Args:
        data_dict (dict): Dictionary mapping CPT names to DataFrames.
        lithology_column (str): Column name of lithology classification.
        save_folder (str): Path to save the figure.
    """
    fixed_lithology_colors = {
        '1': 'blue',
        '2': 'lightblue',
        '2a': 'green',
        '2b': 'yellowgreen',
        '3': 'red',
        '4': 'purple',
        '5': 'orange',
        '6': 'cyan',
        '7': 'magenta',
        '8': 'gray',
        '9': 'black'
    }

    if not data_dict:
        print("[ERROR] No data to plot.")
        return

    n_cpts = len(data_dict)
    fig, axs = plt.subplots(1, n_cpts, figsize=(3 * n_cpts, 10))
    if n_cpts == 1:
        axs = [axs]

    for ax, (cpt_id, df) in zip(axs, data_dict.items()):
        if 'Depth (sbb) [m]' not in df.columns or lithology_column not in df.columns:
            print(f"[WARNING] Missing required columns in {cpt_id}, skipping.")
            continue

        depth = df[depth_column]
        lithology = df[lithology_column].astype(str)

        colors = [fixed_lithology_colors.get(code, 'white') for code in lithology]

        ax.scatter(
            np.zeros_like(depth),
            depth,
            c=colors,
            marker='_',
            s=600,
            linewidths=0.5,
        )

        min_depth, max_depth = depth.min(), depth.max()
        yticks = np.arange(np.floor(min_depth), np.ceil(max_depth) + 0.5, 0.5)
        ax.set_yticks(yticks)

        ax.set_title(cpt_id.replace('_interpreted', ''), fontsize=10)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.grid(True, axis='y', which='major')
        ax.invert_yaxis()

    plt.suptitle(f"{lithology_column} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=8)
               for lith, color in fixed_lithology_colors.items()]
    labels = [f"Zone {lith}" for lith in fixed_lithology_colors.keys()]
    fig.legend(handles, labels, title="Zones", bbox_to_anchor=(1.05, 0.5), loc='center left')

    lithology_clean = re.sub(r"[^\w\-]", "_", lithology_column)
    save_path = os.path.join(save_folder, f"{lithology_clean}_vs_{depth_column}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Lithology plot saved to {save_path}")


def plot_lithology_and_parameters(data_dict, cpt_id, save_folder, lithology_column, parameters, label_dict=None,
                                  layering_df=None):
    """
    Create a horizontal subplot figure for a single CPT:
    - First panel: lithology with legend
    - Remaining panels: parameters vs depth with legends
    - Adds horizontal reference lines from layering_df

    Args:
        data_dict (dict): Dictionary mapping CPT IDs to DataFrames.
        cpt_id (str): The CPT ID to plot.
        save_folder (str): Directory to save the figure.
        lithology_column (str): Column name for lithology classification.
        parameters (list): List of parameter names to plot against depth.
        label_dict (dict, optional): Mapping of parameter names to display labels.
        layering_df (pd.DataFrame, optional): DataFrame with horizontal reference lines for the CPT.

    Returns:
        None
    """
    df = data_dict.get(cpt_id)
    if df is None:
        print(f"[WARNING] CPT {cpt_id} not found.")
        return

    if 'Depth (sbb) [m]' not in df.columns or lithology_column not in df.columns:
        print(f"[WARNING] Missing required columns in {cpt_id}.")
        return

    depth = df['Depth (sbb) [m]']
    n_panels = 1 + len(parameters)
    fig, axs = plt.subplots(1, n_panels, figsize=(3 * n_panels, 10), sharey=True)

    if n_panels == 1:
        axs = [axs]

    # Parse cpt_id to get cpt_name and site
    try:
        cpt_name, site = cpt_id.split("_")[:2]
    except ValueError:
        cpt_name, site = "", ""

    # Extract horizontal line depths from layering_df
    horiz_depths = []
    if layering_df is not None:
        match = layering_df[
            (layering_df["cpt_name"].str.lower() == cpt_name.lower()) &
            (layering_df["site"].str.lower() == site.lower())
            ]
        if not match.empty:
            horiz_depths = [float(val.strip()) for val in match["horiz_lines"].iloc[0].split(",")]

    # --- Lithology colors ---
    lith_colors = {
        '1': 'blue', '2': 'lightblue', '2a': 'green', '2b': 'yellowgreen',
        '3': 'red', '4': 'purple', '5': 'orange', '6': 'cyan',
        '7': 'magenta', '8': 'gray', '9': 'black'
    }

    # --- Lithology subplot (axs[0]) ---
    lith = df[lithology_column].astype(str)
    color_values = [lith_colors.get(code, 'white') for code in lith]
    axs[0].scatter(np.zeros_like(depth), depth, c=color_values, marker='_', s=600, linewidths=0.5)

    axs[0].set_title("Lithology", fontsize=10)
    axs[0].set_xticks([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Depth (m)")
    axs[0].grid(True, axis='y')
    axs[0].invert_yaxis()
    yticks = np.arange(np.floor(depth.min()), np.ceil(depth.max()) + 0.5, 0.5)
    axs[0].set_yticks(yticks)

    # Horizontal lines on lithology panel
    for h in horiz_depths:
        axs[0].axhline(h, color='black', linestyle='--', linewidth=0.8)

    # --- Lithology legend (only present zones) ---
    unique_zones = sorted(set(lith))
    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=lith_colors.get(zone, 'white'),
               label=f"Zone {zone}", markersize=8, linestyle='')
        for zone in unique_zones if zone in lith_colors
    ]
    axs[0].legend(handles=legend_elements, fontsize=7, loc='upper left', title="Zones", title_fontsize=8)

    # --- Parameter subplots with legends ---
    for i, param in enumerate(parameters):
        ax = axs[i + 1]
        if param not in df.columns:
            ax.set_visible(False)
            continue

        values = df[param]
        label = label_dict.get(param, param) if label_dict else param
        ax.plot(values, depth, color='tab:blue', lw=1.5, label=label)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(True, axis='y')
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="best")

        # Add same horizontal lines
        for h in horiz_depths:
            ax.axhline(h, color='black', linestyle='--', linewidth=0.8)

    # --- Save figure ---
    fig.suptitle(f"{cpt_id}", fontsize=12, y=1.02)
    plt.tight_layout()
    clean_id = re.sub(r"[^\w\-]", "_", cpt_id)
    fname = f"{clean_id}_profile.png"
    save_path = os.path.join(save_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved horizontal profile for {cpt_id} → {save_path}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_lithology_by_distance(data_dict, metadata_df, depth_column, lithology_column, layering_df, save_path):
    """
    Plot all CPTs in a single subplot with lithology vs depth, spaced by horizontal distance.
    Overlay black dots at manually defined layer boundaries from layering_df.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fixed_lithology_colors = {
        '1': 'blue', '2': 'lightblue', '2a': 'green', '2b': 'yellowgreen',
        '3': 'red', '4': 'purple', '5': 'orange', '6': 'cyan',
        '7': 'magenta', '8': 'gray', '9': 'black'
    }

    if not data_dict or metadata_df.empty:
        print("[ERROR] No data or metadata to plot.")
        return

    base_coords = metadata_df.iloc[0][['E', 'N']]
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    for _, row in metadata_df.iterrows():
        base_name = row["name"]
        site = row["site"]
        matching_key = next((k for k in data_dict if k.startswith(base_name)), None)
        if not matching_key:
            print(f"[SKIP] No match for {base_name}")
            continue

        df = data_dict[matching_key]
        if depth_column not in df.columns or lithology_column not in df.columns:
            print(f"[SKIP] {matching_key}: missing columns")
            continue

        depth = df[depth_column].dropna()
        lith = df[lithology_column].astype(str)
        if depth.empty:
            print(f"[SKIP] {matching_key}: empty depth")
            continue

        colors = [fixed_lithology_colors.get(z, 'white') for z in lith]
        dist = np.sqrt((row["E"] - base_coords["E"]) ** 2 + (row["N"] - base_coords["N"]) ** 2)

        ax.scatter(np.full_like(depth, dist), depth, c=colors, marker='_', s=600, linewidths=0.5)
        ax.text(dist, depth.min() - 1, base_name, ha='center', fontsize=8, rotation=90)

        # Draw black dots at manual layer boundaries (converted to depth_to_reference)
        layer_match = layering_df[
            (layering_df['site'].str.lower() == site.lower()) &
            (layering_df['cpt_name'].str.lower() == base_name.lower())
            ]
        if not layer_match.empty:
            try:
                # Parse depths in sbb reference (positive down from surface)
                manual_depths = [float(d.strip()) for d in layer_match.iloc[0]['horiz_lines'].split(',')]

                # Get surface elevation from metadata
                elev_row = metadata_df[metadata_df['name'].str.lower() == base_name.lower()]
                if not elev_row.empty:
                    elev = elev_row.iloc[0]['elev_cpt']
                    converted_depths = [elev - d for d in manual_depths]  # Now in depth_to_reference
                    ax.scatter(np.full(len(converted_depths), dist), converted_depths, color='black', s=20, zorder=10)
            except Exception as e:
                print(f"[WARNING] Failed to convert layer depths for {base_name}: {e}")

    # Format axis
    ax.set_xlabel("Distance from first CPT (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{lithology_column} vs Distance", fontsize=14)
    #ax.invert_yaxis()
    ax.grid(axis='y')

    # Legend
    handles = [Line2D([0], [0], color=color, lw=4, label=f"Zone {code}")
               for code, color in fixed_lithology_colors.items()]
    handles.append(Line2D([0], [0], marker='o', color='black', linestyle='', label="Layer boundary", markersize=5))
    ax.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved lithology-by-distance plot → {save_path}")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_lithology_simple_by_distance(data_dict, metadata_df, depth_column, lithology_column, save_path):
    """
    Plot simplified lithology zones across CPTs as a function of horizontal distance and depth.

    Args:
        data_dict (dict): Dictionary of CPT data keyed by file names.
        metadata_df (pd.DataFrame): DataFrame with at least columns ['name', 'E', 'N'].
        depth_column (str): Name of depth column in each CPT file.
        lithology_column (str): Name of lithology column in each CPT file.
        save_path (str): Path to save the output figure.
    """

    lithology_merge_map = {
        '1': 'zone 1',
        '2': 'zone 2a',  # if needed
        '2a': 'zone 2a',
        '2b': 'zone 2b',
        '3': 'zone 3-4',
        '4': 'zone 3-4',
        '5': 'zone 5-6-7',
        '6': 'zone 5-6-7',
        '7': 'zone 5-6-7',
        '8': 'zone 8-9',
        '9': 'zone 8-9'
    }

    combined_zone_colors = {
        "zone 1": "black",
        "zone 2a": "brown",
        "zone 2b": "lightgreen",
        "zone 3-4": "green",
        "zone 5-6-7": "gold",
        "zone 8-9": "grey"
    }

    if not data_dict or metadata_df.empty:
        print("[ERROR] No data or metadata to plot.")
        return

    base_coords = metadata_df.iloc[0][['E', 'N']]
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    for _, row in metadata_df.iterrows():
        base_name = row["name"]
        matching_key = next((k for k in data_dict if k.startswith(base_name)), None)
        if not matching_key:
            print(f"[SKIP] No match for {base_name}")
            continue

        df = data_dict[matching_key]
        if depth_column not in df.columns or lithology_column not in df.columns:
            print(f"[SKIP] {matching_key}: missing columns")
            continue

        depth = df[depth_column].dropna()
        raw_lith = df[lithology_column].astype(str)
        if depth.empty or raw_lith.empty:
            print(f"[SKIP] {matching_key}: empty depth or lithology")
            continue

        # Map raw lithology codes to simplified zones
        simplified_zones = [lithology_merge_map.get(z, None) for z in raw_lith]
        valid_mask = [z is not None for z in simplified_zones]

        depth = depth[valid_mask]
        colors = [combined_zone_colors[simplified_zones[i]] for i, valid in enumerate(valid_mask) if valid]

        dist = np.sqrt((row["E"] - base_coords["E"]) ** 2 + (row["N"] - base_coords["N"]) ** 2)

        ax.scatter(np.full_like(depth, dist), depth, c=colors, marker='_', s=600, linewidths=0.5)
        ax.text(dist, depth.min() - 1, base_name, ha='center', fontsize=8, rotation=90)

    ax.set_xlabel("Distance from first CPT (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Simplified Lithology vs Distance", fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='y')

    # Legend
    handles = [Line2D([0], [0], color=color, lw=4, label=label)
               for label, color in combined_zone_colors.items()]
    ax.legend(handles=handles, title="Merged Zones", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved simplified lithology-by-distance plot → {save_path}")
