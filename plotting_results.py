import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_one_parameter(csv_folder_path, parameter, save_folder):
    """
    Plot a specified parameter vs depth for all CSVs in a folder, one subplot per CPT.
    Each subplot keeps independent X and Y axes.

    Args:
        csv_folder_path (str): Path to folder containing interpreted CPT CSVs.
        parameter (str): Column name to plot (e.g., 'qt (kPa)', 'Vs Robertson (m/s)', etc.)
        save_folder (str): Path where the final figure will be saved.
    """

    # Find all CSVs
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    n_cpts = len(csv_files)
    fig, axs = plt.subplots(1, n_cpts, figsize=(3 * n_cpts, 6))  # No sharey anymore

    # If only one file, axs is not a list
    if n_cpts == 1:
        axs = [axs]

    for ax, csv_file in zip(axs, csv_files):
        # Read CPT CSV
        df = pd.read_csv(csv_file)

        if parameter not in df.columns:
            print(f"[WARNING] {parameter} not found in {os.path.basename(csv_file)}, skipping.")
            continue

        depth = df['Depth* (m)']
        param_values = df[parameter]

        # Plot
        ax.plot(param_values, depth, lw=1)

        ax.set_xlabel(parameter, fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_title(os.path.basename(csv_file).replace('_interpreted.csv', ''), fontsize=8)

        ax.grid(True)

        #invert y axis
        ax.invert_yaxis()

    plt.suptitle(f"{parameter} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    # Clean parameter name for filename
    parameter_clean = re.sub(r"[^\w\-]", "_", parameter)

    save_path = os.path.join(save_folder, f"{parameter_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to {save_path}")






csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
save_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"

plot_all_one_parameter(csv_folder, 'qt (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'fs* (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'Fr (%)', save_path)
plot_all_one_parameter(csv_folder, 'Qtn (kPa)', save_path)
