import pandas as pd
from plotting import load_all_csvs, plot_multi_param_vs_depth, plot_multi_param_with_scatter, plot_lithology_columns
from plotting import plot_lithology_and_parameters


def run_all_plots_for_folder(csv_folder, save_folder, label_dict):
    data_dict = load_all_csvs(csv_folder)

    # Single parameter plots
    single_param_list = [
        'qc (sbb) [kPa]', 'fs (sbb) [kPa]', 'Rf (sbb) [%]',
        'Fr [%]', 'qt [kPa]', 'qn [kPa]', 'Qtn [kPa]', 'PWP_u2 (sbb) [kPa]'
    ]
    for param in single_param_list:
        plot_multi_param_vs_depth(data_dict, [param], save_folder, label_dict)

    # Double parameter plots
    double_param_list = [
        ['relative_density (sbb) [%]', 'relative_density (Robertson and Cabal 2014) [%]'],
        ['Bq (sbb) [-]', 'Bq (Robertson and Cabal 2014) [-]'],
        ['Nkt_Fr [-]', 'Nkt_Bq [-]']
    ]
    for param_pair in double_param_list:
        plot_multi_param_vs_depth(data_dict, param_pair, save_folder, label_dict)

    # Triple parameter plots
    triple_param_list = [
        ['rho (Lengkeek 2022) [kg/m3]', 'rho_peat (Fig8 Lengkeek 2022) [kg/m3]',
         'rho_Gs (Lengkeek+Robertson 2010) [kg/m3]'],
        ['Su (sbb) [kPa]', 'Su (Nkt_Fr) [kPa]', 'Su (Nkt_Bq) [kPa]'],
        ['St (sbb) [-]', 'St (Nkt_Fr) [-]', 'St (Nkt_Bq) [-]']
    ]
    for param_triplet in triple_param_list:
        plot_multi_param_vs_depth(data_dict, param_triplet, save_folder, label_dict)

    # Stress-related parameters
    stress_list = ['PWP_u0 [kPa]', 'sigma_v_prime [kPa]', 'sigma_v_total (Lengkeek 2022) [kPa]']
    plot_multi_param_vs_depth(data_dict, stress_list, save_folder, label_dict)

    # Vs group with scatter overlay
    vs_list = [
        'Vs (Robertson and Cabal 2014) [m/s]', 'Vs (Mayne 2007) [m/s]',
        'Vs (Zhang and Tong 2017) [m/s]', 'Vs (Ahmed 2017) [m/s]',
        'Vs (Kruiver et al 2020) [m/s]'
    ]
    plot_multi_param_with_scatter(data_dict,
                                  vs_list,
                                  scatter_x_col='Vs from SCPTu [m/s]',
                                  scatter_y_col='Z from SCPTu [m/s]',
                                  save_folder=save_folder,
                                  label_dict=label_dict)

    # Lithology
    plot_lithology_columns(data_dict, 'lithology (Robertson and Cabal 2010)', save_folder)
    plot_lithology_columns(data_dict, 'lithology (Lengkeek 2024)', save_folder)

    # Composite plots per CPT
    params_to_plot = [
        'rho (Lengkeek 2022) [kg/m3]', 'Vs (Ahmed 2017) [m/s]',
        'psi (Plewes et al 1992) [-]', 'Su (sbb) [kPa]',
        'qc (sbb) [kPa]', 'Rf (sbb) [%]',
    ]
    for cpt_id in data_dict:
        plot_lithology_and_parameters(
            data_dict, cpt_id, save_folder,
            lithology_column='lithology (Lengkeek 2024)',
            parameters=params_to_plot,
            label_dict=label_dict,
            layering_df=layering_df
        )


plot_labels = {
    'Depth (sbb) [m]': "Depth [m]",
    'Depth_to_reference [m]': "Depth to ref [m]",

    'qc (sbb) [kPa]': "qc [kPa]",
    'fs (sbb) [kPa]': "fs [kPa]",
    'Rf (sbb) [%]': "Rf [%]",
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

# Paths
bavois_results_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois"
chavornay_results_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\chavornay"
ependes_results_folder = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\ependes"
layering_csv_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\layering.csv"

layering_df = pd.read_csv(layering_csv_path)

folders = [bavois_results_folder, chavornay_results_folder, ependes_results_folder]

for folder in folders:
    run_all_plots_for_folder(folder, folder, plot_labels)
