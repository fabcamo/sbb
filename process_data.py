import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod, ShearWaveVelocityMethod, OCRMethod


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


def calc_qn(cpt: GefCpt) -> np.ndarray:
    """
    Calculate the qn value based on the CPT data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.

    returns:
        np.ndarray: The calculated qn value.
    """
    # Calculate qn
    qn = (cpt.qt - cpt.total_stress)
    # Set negative values to 0
    qn[qn < 0] = 0

    return qn


def calc_Bq(cpt: GefCpt) -> np.ndarray:
    """
    Calculate the Bq value based on the CPT data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.

    returns:
        np.ndarray: The calculated Bq value.
    """
    # Calculate Bq
    # Remember to multiply by 1000 to convert from MPa to kPa
    qn = calc_qn(cpt)
    qn_safe = np.maximum(qn, 1e-6)  # Avoid division by zero
    Bq = (cpt.pore_pressure_u2 * 1000 - cpt.hydro_pore_pressure) / qn_safe
    return Bq


def calc_Nkt(cpt: GefCpt) -> float:
    """
    Calculate Nkt based on Fr and Bq according to Robertson (2012) and Mayne & Peuchen (2022).

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.

    Returns:
        Nkt_Fr (float): The calculated Nkt value based on Fr.
        Nkt_Bq (float): The calculated Nkt value based on Bq.
    """
    # 1. Calculate Nkt based on Fr (Robertson, 2012)
    # Make sure Fr is not zero or negative for log calculation
    Fr_safe = np.where(cpt.Fr <= 0, 0.01, cpt.Fr)  # Avoid log(0)
    Nkt_Fr = 10.5 + 7 * np.log10(Fr_safe)

    # 2. Calculate Nkt based on Bq (Mayne and Peuchen, 2022)
    # You need to calculate Bq first if it's not given
    Bq = calc_Bq(cpt)
    # Make sure Bq is not negative or zero for log
    Bq_safe = np.where(Bq + 0.1 <= 0, 0.01, Bq + 0.1)
    Nkt_Bq = 10.5 - 4.6 * np.log(Bq_safe)

    return Nkt_Fr, Nkt_Bq


def calc_Su(cpt: GefCpt, Nkt) -> float:
    """
    Calculate the undrained shear strength (Su) based on the Nkt and the cpt data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.
        Nkt (float): The Nkt value.

    returns:
        Su (float): The calculated undrained shear strength.
    """
    # Calculate Su
    Su = calc_qn(cpt) / Nkt

    return Su

def calc_St(cpt: GefCpt, Nkt) -> np.ndarray:
    """
    Correct calculation of Sensitivity-St using dynamic Nkt and fs.

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.
        Nkt (array or scalar): Calculated Nkt.

    Returns:
        np.ndarray: Sensitivity values.
    """
    qn = calc_qn(cpt)
    fs = np.maximum(cpt.friction, 1e-6)  # Avoid zero or very small fs
    with np.errstate(divide='ignore', invalid='ignore'):
        St = qn / (Nkt * fs)

    return St


def calc_psi(cpt):
    """
    Calculate the Psi (state parameter) for liquefaction assessment from CPTu data.

    Params:
        cpt: CPT object with necessary attributes (qt, friction_nbr (Fr), total_stress, effective_stress, pore_pressure_u2, hydro_pore_pressure).

    Returns:
        psi: Array of Psi values (dimensionless).
    """

    # Safety: avoid division by zero later
    epsilon = 1e-6

    # Step 1: lambda from Fr (%)
    Fr_safe = np.where(cpt.friction_nbr <= 0, epsilon, cpt.friction_nbr)
    lambda_ = Fr_safe / 10

    # Step 2: m from lambda
    m = 11.9 + 13.3 * lambda_

    # Step 3: Estimate phi' from qt and effective stress
    qt_over_sigma_v_eff = np.where(cpt.effective_stress > epsilon, (cpt.qt / cpt.effective_stress), np.nan)
    phi_deg = np.degrees(np.arctan(0.1 + 0.38 * np.log10(np.maximum(qt_over_sigma_v_eff, epsilon))))
    phi_rad = np.radians(phi_deg)

    # Step 4: M from phi
    M = (6 * np.sin(phi_rad)) / (3 - np.sin(phi_rad))

    # Step 5: k from lambda and M
    k = (3 + (0.85 / np.maximum(lambda_, epsilon))) * M

    # Step 6: Qp
    Bq = calc_Bq(cpt)
    Qp = ((cpt.qt - cpt.total_stress) / (cpt.effective_stress + epsilon)) * (1 - Bq)

    # Step 7: Psi calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        psi = -np.log(np.maximum(Qp / np.maximum(k, epsilon), epsilon)) / np.maximum(m, epsilon)

    return psi


def save_results_as_csv(cpt, cpt_dict, results_path):
    """
    After manually populating the CPT object, pre-procesing and interpreting the data, this function
    gathers all the relevant information and saves it as a CSV file for easy access

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.
        cpt_dict (dict): The dictionary containing the CPT data that is read from the pickle file.
        results_path (str): The path where the CSV file will be saved.

    Returns:
        None
    """

    cpt_key = cpt.name.split("_")[0]  # Extract CPTU01 from CPTU01_bavois

    # Calculate Nkt values
    Nkt_Fr, Nkt_Bq = calc_Nkt(cpt)

    Nkt_prov = cpt_dict.get(cpt_key, {}).get('su', np.nan)

    # Create a DataFrame with the interpreted values
    interpreted_data = {
        'Depth* (m)': cpt.depth,
        'Depth to reference level (m)': cpt.depth_to_reference,

        'qc* (kPa)': cpt.tip,
        'fs* (kPa)': cpt.friction,
        'Rf* (%)': cpt.friction_nbr,
        'Fr (%)': cpt.Fr,

        'PWP u2* (kPa)': cpt.pore_pressure_u2 * 1000,
        'PWP u0 (kPa)': cpt.hydro_pore_pressure,

        'Effective Stress (kPa)': cpt.effective_stress,
        'Total Stress (kPa)': cpt.total_stress,

        'qt (kPa)': cpt.qt,
        'qn (kPa)': calc_qn(cpt),
        'Qtn (kPa)': cpt.Qtn,

        'Bq provided (-)': cpt_dict.get(cpt_key, {}).get('Bq', np.nan),
        'Bq calc (-)': calc_Bq(cpt),

        'Nkt {Fr} (-)': Nkt_Fr,
        'Nkt {Bq} (-)': Nkt_Bq,
        'Su provided (kPa)': cpt_dict.get(cpt_key, {}).get('su', np.nan),
        'Su {Fr} (kPa)': calc_Su(cpt, Nkt_Fr),
        'Su {Bq} (kPa)': calc_Su(cpt, Nkt_Bq),

        'St provided (-):': cpt_dict.get(cpt_key, {}).get('St', np.nan),
        'St (Nkt Fr) (-)': calc_St(cpt, Nkt_Fr),
        'St (Nkt Bq) (-)': calc_St(cpt, Nkt_Bq),

        'psi (-)': calc_psi(cpt),
        'psi dGeolib+ (-)': cpt.psi,

        'Vs (m/s)': cpt.vs,

        'E0 (MPa)': cpt.E0,
        'G0 (MPa)': cpt.G0,
        'IC ': cpt.IC,
        'lithology': cpt.lithology,
        'poisson:': cpt.poisson,
        'relative density': cpt.relative_density,
        'rho (kg/m3)': cpt.rho,


    }
    interpreted_df = pd.DataFrame(interpreted_data)
    # Save the DataFrame as a CSV file
    interpreted_df.to_csv(f"{results_path}/{cpt.name}_interpreted.csv", index=False)


# Load the pickle files with the CPT data
bavois_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT\cpt_bavois.pkl"
with open(bavois_pickle, "rb") as f:
    cpt_bavois = pickle.load(f)

chavornay_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT\cpt_chavornay.pkl"
with open(chavornay_pickle, "rb") as f:
    cpt_chavornay = pickle.load(f)

ependes_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_33000-37500_Ependes-Yverdon site investigation\CPT Roh-Daten\cpt_ependes.pkl"
with open(ependes_pickle, "rb") as f:
    cpt_ependes = pickle.load(f)

# Load the metadata for the CPTs (I manually built this)
metadata_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\cpt_other_details.csv"
metadata_df = pd.read_csv(metadata_path)

# Save results path
results_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations"

# random_cpt_path = r"C:\Users\camposmo\Stichting Deltares\RESET - Documents\Activities year 4\2a - Strength modeling\data\sotec\north\24SP1888_DKMP057.GEF"
# cpt = GefCpt()
# cpt.read(random_cpt_path)

# Initialize and populate the CPT objects
cpt_bavois_list = initialize_cpt_objects(cpt_bavois, metadata_df, site_name="bavois")
cpt_chavornay_list = initialize_cpt_objects(cpt_chavornay, metadata_df, site_name="chavornay")
cpt_ependes_list = initialize_cpt_objects(cpt_ependes, metadata_df, site_name="ependes")

# Define the DGeoLib+ interpreter conditions
interpreter = RobertsonCptInterpretation()
interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK2022
interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
interpreter.ocrmethod = OCRMethod.MAYNE
interpreter.user_defined_water_level = True

for cpt in cpt_bavois_list:
    # pre-process the CPT
    cpt.pre_process_data()
    # Run the interpretation
    cpt.interpret_cpt(interpreter)
    # Save the results as a CSV file
    save_results_as_csv(cpt, cpt_bavois, results_path)

for cpt in cpt_chavornay_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.interpret_cpt(interpreter)

for cpt in cpt_ependes_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.interpret_cpt(interpreter)

# cpt = cpt_chavornay_list[1]  # Select the first CPT for demonstration
#
# # Plot in the same space the effective stress and total stress in x vs depth in y
# plt.figure(figsize=(5, 10))
# plt.plot(cpt.effective_stress, cpt.depth, label='Effective Stress', color='darkred')
# plt.plot(cpt.total_stress, cpt.depth, label='Total Stress', color='black')
# plt.plot(cpt.hydro_pore_pressure, cpt.depth, label='Pore Pressure', color='darkblue')
# plt.plot(cpt.pore_pressure_u2 * 1000, cpt.depth, label='Pore Pressure U2', alpha=0.4, color='grey')
# plt.xlabel('Stress (kPa)')
# plt.ylabel('Depth NAP (m)')
# plt.title(f'Total and effective stress - CPT{cpt.name}')
# plt.legend()
# plt.grid(True)
# plt.gca().invert_yaxis()  # Invert y-axis to have depth increasing downwards
# # limit x from 0 to 200
# plt.xlim(-5, 200)
#
# plt.show()
#


# # Create manually the objects for the CPT CPTU01 in bavois
# cpt = GefCpt()
#
# cpt.penetration_length = cpt_bavois['CPTU01']['z']
# cpt.depth = cpt_bavois['CPTU01']['z']
# cpt.tip = cpt_bavois['CPTU01']['qc']
# cpt.friction = cpt_bavois['CPTU01']['fs']
# cpt.pore_pressure_u2 = cpt_bavois['CPTU01']['u2']
# cpt.friction_nbr = cpt_bavois['CPTU01']['|Rf|']
#
# # Get the row that contains the information for CPTU01 and bavois
# row = metadata_df.query("site == 'bavois' and name == 'CPTU01'")
# # Fill other attributes from the metadata
# cpt.name = f"{row['name'].item()}_{row['site'].item()}" if not row.empty else None
# cpt.a = row["a"].item() if not row.empty else None
# cpt.coordinates = [row["E"].item(), row["N"].item()] if not row.empty else [None, None]
# cpt.local_reference_level = -row["elev_cpt"].item() if not row.empty else None
#
##
# cpt.pre_process_data()


# # Loop over each CPT and generate plots
# for cpt_name, data in cpt_bavois.items():
#     z = data.get('z')
#     qc = data.get('qc')
#     fs = data.get('fs')
#
#     fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
#
#     # Plot qc
#     axs[0].plot(qc, z, color='tab:blue')
#     axs[0].set_title(f"{cpt_name} - qc")
#     axs[0].set_xlabel("qc [MN/m²]")
#     axs[0].set_ylabel("Depth z [m]")
#     axs[0].grid(True)
#     axs[0].set_ylim([max(z), min(z)])  # Invert y-axis
#
#     # Plot fs
#     axs[1].plot(fs, z, color='tab:red', linestyle='--')
#     axs[1].set_title(f"{cpt_name} - fs")
#     axs[1].set_xlabel("fs [MN/m²]")
#     axs[1].grid(True)
#     axs[1].set_ylim([max(z), min(z)])  # Invert y-axis
#
#     plt.tight_layout()
#
#     plt.show()
#     plt.close()
