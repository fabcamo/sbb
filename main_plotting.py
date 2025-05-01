from pathlib import Path

from IPython.terminal.shortcuts.auto_match import double_quote

from plotting import load_all_csvs, plot_multi_param_vs_depth

# Paths
csv_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
save_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\test"

# Load all CSVs for the Bavois site
data_dict = load_all_csvs(csv_folder)

plot_labels = {
    'Depth (sbb) [m]': "Depth [m]",
    'Depth_to_reference [m]': "Depth to ref [m]",

    'qc (sbb) [kPa]': "qc [kPa]",
    'fs (sbb) [kPa]': "fs [kPa]",
    'Rf (sbb) [kPa]': "Rf [kPa]",
    'Fr [%]': "Fr [%]",

    'PWP_u2 (sbb) [kPa]': "u₂ [kPa]",
    'PWP_u0 [kPa]': "u₀ [kPa]",

    'sigma_v_prime [kPa]': "σ′v [kPa]",
    'sigma_v_total (Lengkeek 2022) [kPa]': "σv total [kPa]",

    'rho (Lengkeek 2022) [kg/m3]': "ρ (Lengkeek 2022) [kg/m³]",
    'rho_peat (Fig8 Lengkeek 2022) [kg/m3]': "ρ peat (Fig8 Lengkeek 2022) [kg/m³]",
    'rho_Gs (Lengkeek+Robertson 2010) [kg/m3]': "ρ (Gs) (Lengkeek+Robertson 2010) [kg/m³]",

    'relative_density (sbb) [%]': "ID (sbb) [%]",
    'relative_density (Robertson and Cabal 2014) [%]': "ID (Robertson and Cabal 2014) [%]",

    'qt [kPa]': "qt [kPa]",
    'qn [kPa]': "qn [kPa]",
    'Qtn [kPa]': "Qtn [kPa]",

    'IC': "IC",
    'IB': "IB",

    'lithology (Robertson and Cabal 2010)': "Lithology (Robertson)",
    'lithology (Lengkeek 2024)': "Lithology (Lengkeek)",

    'Bq (sbb) [-]': "Bq [-]",
    'Bq (Robertson and Cabal 2014) [-]': "Bq (Robertson and Cabal 2014) [-]",

    'Nkt_Fr [-]': "Nkt (Fr) [-]",
    'Nkt_Bq [-]': "Nkt (Bq) [-]",

    'Su (sbb) [kPa]': "Su [kPa]",
    'Su (Nkt_Fr) [kPa]': "Su (Fr) [kPa]",
    'Su (Nkt_Bq) [kPa]': "Su (Bq) [kPa]",

    'St (sbb) [-]': "St [-]",
    'St (Nkt_Fr) [-]': "St (Fr)",
    'St (Nkt_Bq) [-]': "St (Bq)",

    'psi (Plewes et al 1992) [-]': "ψ (Plewes) [-]",
    'psi (Robertson 2009) [-]': "ψ (Robertson) [-]",

    'Vs (Robertson and Cabal 2014) [m/s]': "Vs (Robertson) [m/s]",
    'Vs (Mayne 2007) [m/s]': "Vs (Mayne) [m/s]",
    'Vs (Zhang and Tong 2017) [m/s]': "Vs (Zhang) [m/s]",
    'Vs (Ahmed 2017) [m/s]': "Vs (Ahmed) [m/s]",
    'Vs (Kruiver et al 2020) [m/s]': "Vs (Kruiver) [m/s]",

    'E0 (Robertson and Cabal 2014) [MPa]': "E₀ (Robertson) [MPa]",
    'E0 (Mayne 2007) [MPa]': "E₀ (Mayne) [MPa]",
    'E0 (Zhang and Tong 2017) [MPa]': "E₀ (Zhang) [MPa]",
    'E0 (Ahmed 2017) [MPa]': "E₀ (Ahmed) [MPa]",
    'E0 (Kruiver et al 2020) [MPa]': "E₀ (Kruiver) [MPa]",

    'G0 (Robertson and Cabal 2014) [MPa]': "G₀ (Robertson) [MPa]",
    'G0 (Mayne 2007) [MPa]': "G₀ (Mayne) [MPa]",
    'G0 (Zhang and Tong 2017) [MPa]': "G₀ (Zhang) [MPa]",
    'G0 (Ahmed 2017) [MPa]': "G₀ (Ahmed) [MPa]",
    'G0 (Kruiver et al 2020) [MPa]': "G₀ (Kruiver) [MPa]",
}


# Plot all of these parameters as single parameters
single_param_list = ['qc (sbb) [kPa]', 'fs (sbb) [kPa]', 'Rf (sbb) [kPa]', 'Fr [%]', 'qt [kPa]', 'qn [kPa]', 'Qtn [kPa]']
# Loop through the list of parameters and plot them as single parameters
for param in single_param_list:
    plot_multi_param_vs_depth(data_dict, [param], save_folder, plot_labels)

# Plot the parameters as double-parameters
double_param_list = [['relative_density (sbb) [%]', 'relative_density (Robertson and Cabal 2014) [%]'],
                     ['Bq (sbb) [-]', 'Bq (Robertson and Cabal 2014) [-]'],
                     ['Nkt_Fr [-]', 'Nkt_Bq [-]']]
# Loop through the list of parameters and plot them as double-parameters
for param_pair in double_param_list:
    plot_multi_param_vs_depth(data_dict, param_pair, save_folder, plot_labels)

# Triple parameter list
triple_param_list = [['rho (Lengkeek 2022) [kg/m3]', 'rho_peat (Fig8 Lengkeek 2022) [kg/m3]', 'rho_Gs (Lengkeek+Robertson 2010) [kg/m3]'],
                     ['Su (sbb) [kPa]', 'Su (Nkt_Fr) [kPa]', 'Su (Nkt_Bq) [kPa]'],
                     ['St (sbb) [-]', 'St (Nkt_Fr) [-]', 'St (Nkt_Bq) [-]']]
# Loop through the list of parameters and plot them as triple-parameters
for param_triplet in triple_param_list:
    plot_multi_param_vs_depth(data_dict, param_triplet, save_folder, plot_labels)

stress_list = ['PWP_u0 [kPa]', 'PWP_u2 (sbb) [kPa]', 'sigma_v_prime [kPa]', 'sigma_v_total (Lengkeek 2022) [kPa]']
plot_multi_param_vs_depth(data_dict, stress_list, save_folder, plot_labels)

