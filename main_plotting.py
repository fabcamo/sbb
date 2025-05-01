from pathlib import Path

from IPython.terminal.shortcuts.auto_match import double_quote

from plotting import load_all_csvs, plot_multi_param_vs_depth

# Paths
csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
save_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\test"

# Load all CSVs for the Bavois site
data_dict = load_all_csvs(csv_folder)

# Plot all of these parameters as single parameters
single_param_list = ['qc (sbb) [kPa]', 'fs (sbb) [kPa]', 'Rf (sbb) [kPa]', 'Fr [%]', 'qt [kPa]', 'qn [kPa]', 'Qtn [kPa]']
# Loop through the list of parameters and plot them as single parameters
for param in single_param_list:
    plot_multi_param_vs_depth(data_dict, [param], save_folder)

# Plot the parameters as multi-parameters
double_param_list = [['relative density (sbb) [%]', 'relative density [%]'],
                     ['Bq (sbb) [-]', 'Bq calc [-]'],
                     ['Nkt (Fr method) [-]', 'Nkt (Bq method) [-]']]