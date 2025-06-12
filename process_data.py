import os
import glob
import pandas as pd
import numpy as np

from geolib_plus.gef_cpt import GefCpt
from calculations import calc_qn, calc_Bq, calc_Nkt, calc_Su, calc_St, calc_psi, calc_IB, L24R10_lithology, filter_by_IB
from calculations import calc_peat_gamma_Lengkeek, calc_gamma_from_Lengkeek_Gs


def create_cpt_object(cpt_name: str, cpt_dict: dict, metadata_df: pd.DataFrame, site_name: str) -> GefCpt:
    """
    Create a GefCpt object and populate it with data from the cpt_dict and metadata_df.

    params:
        cpt_name (str): The name of the CPT.
        cpt_dict (dict): Dictionary containing CPT data.
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
        site_name (str): The name of the site.

    returns:
        GefCpt: An instance of the GefCpt class populated with data.
    """
    cpt = GefCpt()

    data = cpt_dict[cpt_name]
    cpt.penetration_length = data.get('z')
    cpt.depth = data.get('z')
    cpt.tip = data.get('qc')
    cpt.friction = data.get('fs')
    cpt.pore_pressure_u2 = data.get('u2')
    cpt.friction_nbr = data.get('|Rf|')

    # Metadata lookup
    row = metadata_df.query(f"site == '{site_name}' and name == '{cpt_name}'")

    if not row.empty:
        cpt.name = f"{row['name'].item()}_{row['site'].item()}"
        cpt.a = row["a"].item()
        cpt.coordinates = [row["E"].item(), row["N"].item()]
        cpt.local_reference_level = row["elev_cpt"].item()
        cpt.pwp = row["gwl_reports"].item()
    else:
        cpt.name = cpt_name
        cpt.coordinates = [None, None]
        cpt.local_reference_level = None
        cpt.a = None
        cpt.pwp = None

    return cpt


def initialize_cpt_objects(cpt_dict: dict, metadata_df: pd.DataFrame, site_name: str) -> list:
    """
    Initializes all the CPT objects in the given dictionary and populates them using the craete_cpt_object function.

    params:
        cpt_dict (dict): Dictionary containing CPT data.
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
        site_name (str): The name of the site.

    returns:
        list: A list of initialized GefCpt objects.
    """
    cpt_objects = []
    for cpt_name in cpt_dict:
        try:

            cpt_obj = create_cpt_object(cpt_name, cpt_dict, metadata_df, site_name)
            cpt_objects.append(cpt_obj)
        except Exception as e:
            print(f"[WARNING] Skipped {cpt_name} due to error: {e}")
    return cpt_objects


def calc_Poisson_below_gwl(cpt: GefCpt) -> float:
    """
    If the measurement is below the ground water level, replace whatever poisson ratio with 0.495, If it is above,
    keep the original value.

    Params:
        cpt (GefCpt): The CPT object.

    Returns:
        poisson_gwl (float): The Poisson ratio corrected by the ground water level.
    """
    # Get the original Poisson ratio
    poisson = cpt.poisson
    # Get the depth of the CPT
    depth = cpt.depth
    # Get the ground water level
    gwl = cpt.pwp
    # Get the depth to reference
    depth_to_ref = cpt.depth_to_reference

    # For every depth to reference, check if it is above or below the ground water level
    poisson_gwl = np.where(depth_to_ref < gwl, 0.495, poisson)

    return poisson_gwl


def build_interpreted_data(cpt, cpt_dict, vs_results, polygons_L24R10):
    """
    Build the interpreted data dictionary ready for saving.

    Args:
        cpt (GefCpt): The CPT object.
        cpt_dict (dict): The dictionary containing CPT data.
        vs_results (dict): Dictionary with Vs, G0, E0 from different methods.
        polygons_L24R10 (dict): Dictionary with Lengkeek 2024 classification polygons.

    Returns:
        dict: Interpreted data dictionary.
    """
    cpt_key = cpt.name.split("_")[0]

    # Calculate everything needed
    Nkt_Fr, Nkt_Bq = calc_Nkt(cpt)
    Su_Fr = calc_Su(cpt, Nkt_Fr)
    Su_Bq = calc_Su(cpt, Nkt_Bq)
    St_Fr = calc_St(cpt, Nkt_Fr)
    St_Bq = calc_St(cpt, Nkt_Bq)
    psi_manual = calc_psi(cpt)
    IB = calc_IB(cpt)

    # Filter Su, St, Psi
    Su_Fr_filtered, Su_Bq_filtered, St_Fr_filtered, St_Bq_filtered, psi_manual_filtered = \
        filter_by_IB(Su_Fr, Su_Bq, St_Fr, St_Bq, psi_manual, IB)

    lithology_L24R10 = L24R10_lithology(cpt, polygons_L24R10)

    # Build dictionary (same as you had)
    interpreted_data = {
        'Depth (sbb) [m]': cpt.depth,
        'Depth_to_reference [m]': cpt.depth_to_reference,

        'qc (sbb) [kPa]': np.array(cpt.tip),
        'fs (sbb) [kPa]': np.array(cpt.friction),
        'Rf (sbb) [%]': cpt.friction_nbr,
        'Fr [%]': cpt.Fr,

        'PWP_u2 (sbb) [kPa]': cpt.pore_pressure_u2 * 1000,
        'PWP_u0 [kPa]': cpt.hydro_pore_pressure,

        'sigma_v_prime [kPa]': cpt.effective_stress,
        'sigma_v_total (Lengkeek 2022) [kPa]': cpt.total_stress,

        'rho (Lengkeek 2022) [kg/m3]': cpt.rho,
        'rho_peat (Fig8 Lengkeek 2022) [kg/m3]': (calc_peat_gamma_Lengkeek(cpt, lithology_L24R10) * 1000) / cpt.g,
        'rho_Gs (Lengkeek+Robertson 2010) [kg/m3]': (calc_gamma_from_Lengkeek_Gs(cpt) * 1000) / cpt.g,

        'relative_density (sbb) [%]': cpt_dict.get(cpt_key, {}).get('Id', np.nan),
        'relative_density (Robertson and Cabal 2014) [%]': cpt.relative_density,

        'qt [kPa]': cpt.qt,
        'qn [kPa]': calc_qn(cpt),
        'Qtn [kPa]': cpt.Qtn,

        'IC': cpt.IC,
        'IB': IB,

        'lithology (Robertson and Cabal 2010)': cpt.lithology,
        'lithology (Lengkeek 2024)': lithology_L24R10,

        'Bq (sbb) [-]': cpt_dict.get(cpt_key, {}).get('Bq', np.nan),
        'Bq (Robertson and Cabal 2014) [-]': calc_Bq(cpt),

        'Nkt_Fr [-]': Nkt_Fr,
        'Nkt_Bq [-]': Nkt_Bq,

        'Su (sbb) [kPa]': cpt_dict.get(cpt_key, {}).get('su CSS', np.nan),
        'Su (Nkt_Fr) [kPa]': Su_Fr_filtered,
        'Su (Nkt_Bq) [kPa]': Su_Bq_filtered,

        'St (sbb) [-]': cpt_dict.get(cpt_key, {}).get('St', np.nan),
        'St (Nkt_Fr) [-]': St_Fr_filtered,
        'St (Nkt_Bq) [-]': St_Bq_filtered,

        'psi (Plewes et al 1992) [-]': psi_manual_filtered,
        'psi (Robertson 2009) [-]': cpt.psi,

        'Vs (Robertson and Cabal 2014) [m/s]': vs_results["Robertson"]["vs"],
        'Vs (Mayne 2007) [m/s]': vs_results["Mayne"]["vs"],
        'Vs (Zhang and Tong 2017) [m/s]': vs_results["Zang"]["vs"],
        'Vs (Ahmed 2017) [m/s]': vs_results["Ahmed"]["vs"],
        'Vs (Kruiver et al 2020) [m/s]': vs_results["Kruiver"]["vs"],

        'Poisson (Mayne 2007) [-]': cpt.poisson,

        'E0 (Robertson and Cabal 2014) [kPa]': vs_results["Robertson"]["E0"] / 1000,
        'E0 (Mayne 2007) [kPa]': vs_results["Mayne"]["E0"] / 1000,
        'E0 (Zhang and Tong 2017) [kPa]': vs_results["Zang"]["E0"] / 1000,
        'E0 (Ahmed 2017) [kPa]': vs_results["Ahmed"]["E0"] / 1000,
        'E0 (Kruiver et al 2020) [kPa]': vs_results["Kruiver"]["E0"] / 1000,

        'G0 (Robertson and Cabal 2014) [kPa]': vs_results["Robertson"]["G0"] / 1000,
        'G0 (Mayne 2007) [kPa]': vs_results["Mayne"]["G0"] / 1000,
        'G0 (Zhang and Tong 2017) [kPa]': vs_results["Zang"]["G0"] / 1000,
        'G0 (Ahmed 2017) [kPa]': vs_results["Ahmed"]["G0"] / 1000,
        'G0 (Kruiver et al 2020) [kPa]': vs_results["Kruiver"]["G0"] / 1000,

        'Ground water level [m]': cpt.pwp,
        'Poisson ratio gwl [-]': calc_Poisson_below_gwl(cpt),
    }

    return interpreted_data


def save_results_as_csv(cpt, cpt_dict, results_path, vs_results, polygons_L24R10):
    """
    Save interpreted CPT results.

    Args:
        cpt (GefCpt): CPT object.
        cpt_dict (dict): Raw dictionary from pickle.
        results_path (str): Save path.
        vs_results (dict): Vs and stiffnesses dictionary.
        polygons_L24R10 (dict): Lengkeek polygons.
    """

    interpreted_data = build_interpreted_data(cpt, cpt_dict, vs_results, polygons_L24R10)

    interpreted_df = pd.DataFrame(interpreted_data)
    interpreted_df.to_csv(f"{results_path}/{cpt.name}_interpreted.csv", index=False, encoding="utf-8")


def add_measured_vs_data(excel_path, csv_folder_path):
    """
    Add measured Z and Vs data from SCPTu Excel file into the existing interpreted CPT CSVs.

    Args:
        excel_path (str): Path to the Excel file (with multiple sheets).
        csv_folder_path (str): Path to folder containing the interpreted CPT CSVs.
    """
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))
    scptu_files = [f for f in csv_files if "SCPTU" in os.path.basename(f)]

    print(f"[INFO] Found {len(scptu_files)} SCPTU files to process.")

    for csv_file in scptu_files:
        file_name = os.path.basename(csv_file)

        cpt_number = file_name.split("_")[0].replace("SCPTU", "").zfill(2)
        expected_sheet_name = f"SCPTU {cpt_number}-24 - Vs"

        if expected_sheet_name not in sheet_names:
            print(f"[ERROR] Sheet {expected_sheet_name} not found for {file_name}, skipping.")
            continue

        # Read Z and Vs columns
        df_sdmt = pd.read_excel(excel_path, sheet_name=expected_sheet_name, skiprows=3, usecols="A:B")
        df_sdmt.columns = ["Z from SCPTu [m]", "Vs from SCPTu [m/s]"]  # renamed here

        # Read existing interpreted CSV
        df_cpt = pd.read_csv(csv_file, encoding="utf-8")

        # Add the renamed SCPTu columns
        df_cpt["Z from SCPTu [m]"] = df_sdmt["Z from SCPTu [m]"]
        df_cpt["Vs from SCPTu [m/s]"] = df_sdmt["Vs from SCPTu [m/s]"]

        # Overwrite original CSV
        df_cpt.to_csv(csv_file, index=False, encoding="utf-8")

        print(f"[INFO] Added SCPTu data to {file_name}")

    print("[INFO] Finished adding SCPTu measured data.")


def sort_CPT_by_coordinates(cpt_metadata: str, sort_direction: str = "E-W"):
    """
    Reads the CPT metadata from a CSV file and sorts the CPTs based on their coordinates.
    The user can specify the sorting direction as either
    "E-W" (East-West) or "W-E" (West-East), N-S (North-South) or S-N (South-North).

    Args:
        cpt_metadata (str): Path to the CSV file containing CPT metadata.
        sort_direction (str): Direction to sort the CPTs. Options are "E-W", "W-E", "N-S", "S-N".

    Returns:
        metadata_ordered (DataFrame): DataFrame containing the sorted CPT metadata with names.
    """
    # Read the metadata CSV file
    metadata_df = pd.read_csv(cpt_metadata)

    # Sort the dataframe based on the specified direction
    if sort_direction == "E-W":
        sorted_df = metadata_df.sort_values(by="E", ascending=True)
    elif sort_direction == "W-E":
        sorted_df = metadata_df.sort_values(by="E", ascending=False)
    elif sort_direction == "N-S":
        sorted_df = metadata_df.sort_values(by="N", ascending=True)
    elif sort_direction == "S-N":
        sorted_df = metadata_df.sort_values(by="N", ascending=False)
    else:
        raise ValueError("Invalid sort direction. Use 'E-W', 'W-E', 'N-S', or 'S-N'.")

    return sorted_df

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        distance (float): The Euclidean distance between the two points.
    """
    distance = np.sqrt((point1.iloc[0] - point2.iloc[0]) ** 2 + (point1.iloc[1] - point2.iloc[1]) ** 2)
    return distance
