import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt


# This is a file exclusively for visualization of the results

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

        # invert y axis
        ax.invert_yaxis()

    plt.suptitle(f"{parameter} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    # Clean parameter name for filename
    parameter_clean = re.sub(r"[^\w\-]", "_", parameter)

    save_path = os.path.join(save_folder, f"{parameter_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to {save_path}")


def plot_all_two_parameters(csv_folder_path, parameter1, parameter2, save_folder):
    """
    Plot two specified parameters vs depth for all CSVs in a folder, one subplot per CPT.
    Each subplot keeps independent X and Y axes.

    Args:
        csv_folder_path (str): Path to folder containing interpreted CPT CSVs.
        parameter1 (str): First column name to plot.
        parameter2 (str): Second column name to plot.
        save_folder (str): Path where the final figure will be saved.
    """

    # Find all CSVs
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    n_cpts = len(csv_files)
    fig, axs = plt.subplots(1, n_cpts, figsize=(4 * n_cpts, 6))  # slightly wider

    if n_cpts == 1:
        axs = [axs]

    for ax, csv_file in zip(axs, csv_files):
        df = pd.read_csv(csv_file)

        if parameter1 not in df.columns or parameter2 not in df.columns:
            print(f"[WARNING] {parameter1} or {parameter2} not found in {os.path.basename(csv_file)}, skipping.")
            continue

        depth = df['Depth* (m)']
        values1 = df[parameter1]
        values2 = df[parameter2]

        # Plot both parameters
        ax.plot(values1, depth, label=parameter1, lw=1.5)
        ax.plot(values2, depth, label=parameter2, lw=1.5, linestyle='--')

        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_title(os.path.basename(csv_file).replace('_interpreted.csv', ''), fontsize=8)

        ax.grid(True)
        ax.invert_yaxis()

        ax.legend(fontsize=6, loc='best')

    plt.suptitle(f"{parameter1} and {parameter2} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    # Clean parameter names for filename
    param1_clean = re.sub(r"[^\w\-]", "_", parameter1)
    param2_clean = re.sub(r"[^\w\-]", "_", parameter2)

    save_path = os.path.join(save_folder, f"{param1_clean}_and_{param2_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to {save_path}")


def plot_all_three_parameters(csv_folder_path, parameter1, parameter2, parameter3, save_folder):
    """
    Plot three specified parameters vs depth for all CSVs in a folder, one subplot per CPT.
    Each subplot keeps independent X and Y axes.

    Args:
        csv_folder_path (str): Path to folder containing interpreted CPT CSVs.
        parameter1 (str): First column name to plot.
        parameter2 (str): Second column name to plot.
        parameter3 (str): Third column name to plot.
        save_folder (str): Path where the final figure will be saved.
    """

    # Find all CSVs
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    n_cpts = len(csv_files)
    fig, axs = plt.subplots(1, n_cpts, figsize=(4.5 * n_cpts, 6))  # a bit wider

    if n_cpts == 1:
        axs = [axs]

    for ax, csv_file in zip(axs, csv_files):
        df = pd.read_csv(csv_file)

        # Check if all parameters exist
        missing_cols = [p for p in [parameter1, parameter2, parameter3] if p not in df.columns]
        if missing_cols:
            print(f"[WARNING] {missing_cols} not found in {os.path.basename(csv_file)}, skipping.")
            continue

        depth = df['Depth* (m)']
        values1 = df[parameter1]
        values2 = df[parameter2]
        values3 = df[parameter3]

        # Plot all three parameters
        ax.plot(values1, depth, label=parameter1, lw=1.5)
        ax.plot(values2, depth, label=parameter2, lw=1.5, linestyle='--')
        ax.plot(values3, depth, label=parameter3, lw=1.5, linestyle=':')

        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_title(os.path.basename(csv_file).replace('_interpreted.csv', ''), fontsize=8)

        ax.grid(True)
        ax.invert_yaxis()

        ax.legend(fontsize=6, loc='best')

    plt.suptitle(f"{parameter1}, {parameter2} and {parameter3} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    # Clean parameter names for filename
    param1_clean = re.sub(r"[^\w\-]", "_", parameter1)
    param2_clean = re.sub(r"[^\w\-]", "_", parameter2)
    param3_clean = re.sub(r"[^\w\-]", "_", parameter3)

    save_path = os.path.join(save_folder, f"{param1_clean}_{param2_clean}_{param3_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to {save_path}")


def plot_all_four_parameters(csv_folder_path, parameter1, parameter2, parameter3, parameter4, save_folder):
    """
    Plot four specified parameters vs depth for all CSVs in a folder, one subplot per CPT.
    Each subplot keeps independent X and Y axes.

    Args:
        csv_folder_path (str): Path to folder containing interpreted CPT CSVs.
        parameter1 (str): First column name to plot.
        parameter2 (str): Second column name to plot.
        parameter3 (str): Third column name to plot.
        parameter4 (str): Fourth column name to plot.
        save_folder (str): Path where the final figure will be saved.
    """

    # Find all CSVs
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    n_cpts = len(csv_files)
    fig, axs = plt.subplots(1, n_cpts, figsize=(5 * n_cpts, 6))  # wider if 4 lines

    if n_cpts == 1:
        axs = [axs]

    for ax, csv_file in zip(axs, csv_files):
        df = pd.read_csv(csv_file)

        # Check if all parameters exist
        missing_cols = [p for p in [parameter1, parameter2, parameter3, parameter4] if p not in df.columns]
        if missing_cols:
            print(f"[WARNING] {missing_cols} not found in {os.path.basename(csv_file)}, skipping.")
            continue

        depth = df['Depth* (m)']
        values1 = df[parameter1]
        values2 = df[parameter2]
        values3 = df[parameter3]
        values4 = df[parameter4]

        # Plot all four parameters
        ax.plot(values1, depth, label=parameter1, lw=1.5)
        ax.plot(values2, depth, label=parameter2, lw=1.5, linestyle='--')
        ax.plot(values3, depth, label=parameter3, lw=1.5, linestyle=':')
        ax.plot(values4, depth, label=parameter4, lw=1.5, linestyle='-.')

        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_title(os.path.basename(csv_file).replace('_interpreted.csv', ''), fontsize=8)

        ax.grid(True)
        ax.invert_yaxis()

        ax.legend(fontsize=6, loc='best')

    plt.suptitle(f"{parameter1}, {parameter2}, {parameter3} and {parameter4} vs Depth for all CPTs", y=1.02,
                 fontsize=12)
    plt.tight_layout()

    # Clean parameter names for filename
    param_clean = "_".join([re.sub(r"[^\w\-]", "_", p) for p in [parameter1, parameter2, parameter3, parameter4]])

    save_path = os.path.join(save_folder, f"{param_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to {save_path}")


def plot_all_lithology(csv_folder_path, lithology_column, save_folder):
    """
    Plot lithology (discrete classes) vs depth for all CPTs in a folder.

    Args:
        csv_folder_path (str): Path to folder containing interpreted CPT CSVs.
        lithology_column (str): Column name of lithology (e.g., 'lithology Robertson', 'lithology Lengkeek 2024').
        save_folder (str): Path where the final figure will be saved.
    """

    # Fixed color mapping for all known zones
    fixed_lithology_colors = {
        '1': 'blue',
        '2': 'lightblue',  # Only for Robertson
        '2a': 'green',  # Only for Lengkeek
        '2b': 'yellowgreen',  # Only for Lengkeek
        '3': 'red',
        '4': 'purple',
        '5': 'orange',
        '6': 'cyan',
        '7': 'magenta',
        '8': 'gray',
        '9': 'black'
    }

    # Read all CSVs
    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    n_cpts = len(csv_files)
    fig, axs = plt.subplots(1, n_cpts, figsize=(3 * n_cpts, 6))

    if n_cpts == 1:
        axs = [axs]

    for ax, csv_file in zip(axs, csv_files):
        df = pd.read_csv(csv_file)

        if lithology_column not in df.columns:
            print(f"[WARNING] {lithology_column} not found in {os.path.basename(csv_file)}, skipping.")
            continue

        depth = df['Depth* (m)']
        lithology = df[lithology_column]

        # Map colors
        color_values = [fixed_lithology_colors.get(str(l), 'white') for l in lithology]

        ax.scatter(
            np.zeros_like(depth),  # Plot at x=0
            depth,
            c=color_values,
            marker='_',  # small square
            s=600,
            linewidths=0.5,
        )

        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_title(os.path.basename(csv_file).replace('_interpreted.csv', ''), fontsize=8)
        ax.grid(True)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_xlabel('')

    plt.suptitle(f"{lithology_column} vs Depth for all CPTs", y=1.02, fontsize=12)
    plt.tight_layout()

    # Legend
    handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=8)
               for lith, color in fixed_lithology_colors.items()]
    labels = [f"Zone {lith}" for lith in fixed_lithology_colors.keys()]
    fig.legend(handles, labels, title="Zones", bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Save
    lithology_clean = re.sub(r"[^\w\-]", "_", lithology_column)
    save_path = os.path.join(save_folder, f"{lithology_clean}_vs_depth.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Lithology plot saved to {save_path}")



# For BAVOIS

# csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
# save_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
#
# plot_all_one_parameter(csv_folder, 'qt (kPa)', save_path)
# plot_all_one_parameter(csv_folder, 'fs* (kPa)', save_path)
# plot_all_one_parameter(csv_folder, 'Fr (%)', save_path)
# plot_all_one_parameter(csv_folder, 'Qtn (kPa)', save_path)
#
# plot_all_two_parameters(csv_folder, 'Bq provided (-)', 'Bq calc (-)', save_path)
# plot_all_two_parameters(csv_folder, 'Nkt {Fr} (-)', 'Nkt {Bq} (-)', save_path)
# plot_all_two_parameters(csv_folder, 'psi (-)', 'psi dGeolib+ (-)', save_path)
#
# plot_all_three_parameters(csv_folder,
#                           'St provided (-):',
#                           'St (Nkt Fr) (-)',
#                           'St (Nkt Bq) (-)',
#                           save_path)
#
# plot_all_three_parameters(csv_folder,
#                           'Su provided (kPa)',
#                           'Su {Fr} (kPa)',
#                           'Su {Bq} (kPa)',
#                           save_path)
#
# plot_all_four_parameters(csv_folder,
#                           'Vs Robertson (m/s)',
#                           'Vs Mayne (m/s)',
#                           'Vs Zhang (m/s)',
#                          'Vs Ahmed (m/s)',
#                           save_path)
#
# plot_all_four_parameters(csv_folder,
#                           'E0 Robertson (MPa)',
#                           'E0 Mayne (MPa)',
#                           'E0 Zhang (MPa)',
#                          'E0 Ahmed (MPa)',
#                           save_path)
#
# # Lithologies
# plot_all_lithology(csv_folder, 'lithology Robertson', save_path)
# plot_all_lithology(csv_folder, 'lithology Lengkeek 2024', save_path)



# For CHAVORNAY

csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\chavornay"
save_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\chavornay"

plot_all_one_parameter(csv_folder, 'qt (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'fs* (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'Fr (%)', save_path)
plot_all_one_parameter(csv_folder, 'Qtn (kPa)', save_path)

plot_all_two_parameters(csv_folder, 'Bq provided (-)', 'Bq calc (-)', save_path)
plot_all_two_parameters(csv_folder, 'Nkt {Fr} (-)', 'Nkt {Bq} (-)', save_path)
plot_all_two_parameters(csv_folder, 'psi (-)', 'psi dGeolib+ (-)', save_path)

plot_all_three_parameters(csv_folder,
                          'St provided (-):',
                          'St (Nkt Fr) (-)',
                          'St (Nkt Bq) (-)',
                          save_path)

plot_all_three_parameters(csv_folder,
                          'Su provided (kPa)',
                          'Su {Fr} (kPa)',
                          'Su {Bq} (kPa)',
                          save_path)

plot_all_four_parameters(csv_folder,
                          'Vs Robertson (m/s)',
                          'Vs Mayne (m/s)',
                          'Vs Zhang (m/s)',
                         'Vs Ahmed (m/s)',
                          save_path)

plot_all_four_parameters(csv_folder,
                          'E0 Robertson (MPa)',
                          'E0 Mayne (MPa)',
                          'E0 Zhang (MPa)',
                         'E0 Ahmed (MPa)',
                          save_path)

# Lithologies
plot_all_lithology(csv_folder, 'lithology Robertson', save_path)
plot_all_lithology(csv_folder, 'lithology Lengkeek 2024', save_path)


# For EPENDES

csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\ependes"
save_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\ependes"

plot_all_one_parameter(csv_folder, 'qt (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'fs* (kPa)', save_path)
plot_all_one_parameter(csv_folder, 'Fr (%)', save_path)
plot_all_one_parameter(csv_folder, 'Qtn (kPa)', save_path)

plot_all_two_parameters(csv_folder, 'Bq provided (-)', 'Bq calc (-)', save_path)
plot_all_two_parameters(csv_folder, 'Nkt {Fr} (-)', 'Nkt {Bq} (-)', save_path)
plot_all_two_parameters(csv_folder, 'psi (-)', 'psi dGeolib+ (-)', save_path)

plot_all_three_parameters(csv_folder,
                          'St provided (-):',
                          'St (Nkt Fr) (-)',
                          'St (Nkt Bq) (-)',
                          save_path)

plot_all_three_parameters(csv_folder,
                          'Su provided (kPa)',
                          'Su {Fr} (kPa)',
                          'Su {Bq} (kPa)',
                          save_path)

plot_all_four_parameters(csv_folder,
                          'Vs Robertson (m/s)',
                          'Vs Mayne (m/s)',
                          'Vs Zhang (m/s)',
                         'Vs Ahmed (m/s)',
                          save_path)

plot_all_four_parameters(csv_folder,
                          'E0 Robertson (MPa)',
                          'E0 Mayne (MPa)',
                          'E0 Zhang (MPa)',
                         'E0 Ahmed (MPa)',
                          save_path)

# Lithologies
plot_all_lithology(csv_folder, 'lithology Robertson', save_path)
plot_all_lithology(csv_folder, 'lithology Lengkeek 2024', save_path)