import pickle
import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
from shapely.geometry import Point

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


def calculate_vs_with_different_methods(cpt, interpreter):
    """
    Calculate Vs, G0, and E0 using different methods without modifying the original CPT.

    Args:
        cpt (GefCpt): The CPT object.
        interpreter (RobertsonCptInterpretation): The interpreter object.

    Returns:
        dict: Dictionary where keys are method names and values are dicts of vs, G0, E0.
    """
    vs_results = {}

    methods = {
        "Robertson": ShearWaveVelocityMethod.ROBERTSON,
        "Mayne": ShearWaveVelocityMethod.MAYNE,
        "Zang": ShearWaveVelocityMethod.ZANG,
        "Ahmed": ShearWaveVelocityMethod.AHMED,
    }

    for method_name, method_enum in methods.items():
        # Create a deep copy to avoid overwriting the original CPT
        cpt_copy = copy.deepcopy(cpt)

        interpreter.shearwavevelocitymethod = method_enum
        cpt_copy.interpret_cpt(interpreter)

        vs_results[method_name] = {
            "vs": cpt_copy.vs.copy(),
            "G0": cpt_copy.G0.copy(),
            "E0": cpt_copy.E0.copy(),
        }

    return vs_results


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


def L24R10_lithology(cpt, polygons):
    """
    Classify a CPT point-by-point according to custom polygons (Lengkeek 2024).

    Args:
        cpt (GefCpt): CPT object.
        polygons (dict): Dictionary with shapely polygons.

    Returns:
        List: List of classified zone names (e.g., '2a', '5', etc.)
    """
    classifications = []
    for i in range(len(cpt.depth)):
        # Get coordinates
        x_coord = cpt.friction_nbr[i]
        y_coord = cpt.qt[i] / 100

        # Apply limits
        x_coord = np.clip(x_coord, 0.1001, 20)
        y_coord = np.clip(y_coord, 1.01, 999.99)

        point = Point(x_coord, y_coord)

        zone = None
        for zone_name, polygon in polygons.items():
            if polygon.contains(point):
                # Remove 'zone ' if it exists
                cleaned_zone = zone_name.replace('zone ', '').strip()
                zone = cleaned_zone
                break

        classifications.append(zone if zone else None)

    return classifications


def save_results_as_csv(cpt, cpt_dict, results_path, vs_results, polygons_L24R10):
    """
    Save interpreted CPT results including Lengkeek 2024 lithology classification.

    Args:
        cpt (GefCpt): The CPT object.
        cpt_dict (dict): The dictionary containing CPT data.
        results_path (str): Where to save the CSV.
        vs_list (list): List of Vs values (Robertson, Mayne, Zang).
        polygons_L24R10 (dict): Dictionary with Lengkeek 2024 classification polygons.
    """

    cpt_key = cpt.name.split("_")[0]

    # Unpack the calculated Nkt and Vs values
    Nkt_Fr, Nkt_Bq = calc_Nkt(cpt)

    # No need to manually unpack anymore
    vs_robertson = vs_results["Robertson"]["vs"]
    G0_robertson = vs_results["Robertson"]["G0"]
    E0_robertson = vs_results["Robertson"]["E0"]

    vs_mayne = vs_results["Mayne"]["vs"]
    G0_mayne = vs_results["Mayne"]["G0"]
    E0_mayne = vs_results["Mayne"]["E0"]

    vs_zhang = vs_results["Zang"]["vs"]
    G0_zang = vs_results["Zang"]["G0"]
    E0_zang = vs_results["Zang"]["E0"]

    vs_ahmed = vs_results["Ahmed"]["vs"]
    G0_ahmed = vs_results["Ahmed"]["G0"]
    E0_ahmed = vs_results["Ahmed"]["E0"]

    # Classify lithology based on Lengkeek 2024
    lithology_L24R10 = L24R10_lithology(cpt, polygons_L24R10)

    # Build the interpreted data dictionary
    interpreted_data = {
        'Depth* (m)': cpt.depth,
        'Depth to reference level (m)': cpt.depth_to_reference,

        'qc* (kPa)': np.array(cpt.tip),
        'fs* (kPa)': np.array(cpt.friction),
        'Rf* (%)': cpt.friction_nbr,
        'Fr (%)': cpt.Fr,

        'PWP u2* (kPa)': cpt.pore_pressure_u2 * 1000,
        'PWP u0 (kPa)': cpt.hydro_pore_pressure,

        'Effective Stress (kPa)': cpt.effective_stress,
        'Total Stress (kPa)': cpt.total_stress,

        'qt (kPa)': cpt.qt,
        'qn (kPa)': calc_qn(cpt),
        'Qtn (kPa)': cpt.Qtn,

        'IC ': cpt.IC,
        'lithology Robertson': cpt.lithology,
        'lithology Lengkeek 2024': lithology_L24R10,

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

        'Vs Robertson (m/s)': vs_robertson,
        'Vs Mayne (m/s)': vs_mayne,
        'Vs Zhang (m/s)': vs_zhang,
        'Vs Ahmed (m/s)': vs_ahmed,

        'E0 Robertson (MPa)': E0_robertson/1000,
        'E0 Mayne (MPa)': E0_mayne/1000,
        'E0 Zhang (MPa)': E0_zang/1000,
        'E0 Ahmed (MPa)': E0_ahmed/1000,

        'G0 Robertson (MPa)': G0_robertson/1000,
        'G0 Mayne (MPa)': G0_mayne/1000,
        'G0 Zhang (MPa)': G0_zang/1000,
        'G0 Ahmed (MPa)': G0_ahmed/1000,
    }

    interpreted_df = pd.DataFrame(interpreted_data)
    interpreted_df.to_csv(f"{results_path}/{cpt.name}_interpreted.csv", index=False)




#########################################################################################################################

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

#  Paths for the pickel files containing the Lengkeek (2024) classification
L24R10_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\polygons_L24R10.pkl"

# Load the polygons from the pickle file for the L24R10 classification
with open(L24R10_path, "rb") as f:
    polygons_L24R10 = pickle.load(f)

# List of predefined colors for the zones
colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'black', 'grey']


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
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    #save_results_as_csv(cpt, cpt_bavois, r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois', vs_list, polygons_L24R10)

for cpt in cpt_chavornay_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    save_results_as_csv(cpt, cpt_chavornay, r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois', vs_list, polygons_L24R10)


for cpt in cpt_ependes_list:
    # pre-process the CPT
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    save_results_as_csv(cpt, cpt_ependes, r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois', vs_list, polygons_L24R10)

