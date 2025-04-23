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
    cpt.interpret_cpt(interpreter)

for cpt in cpt_chavornay_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.interpret_cpt(interpreter)

for cpt in cpt_ependes_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.interpret_cpt(interpreter)

cpt = cpt_chavornay_list[1]  # Select the first CPT for demonstration

# Plot in the same space the effective stress and total stress in x vs depth in y
plt.figure(figsize=(5, 10))
plt.plot(cpt.effective_stress, cpt.depth, label='Effective Stress', color='darkred')
plt.plot(cpt.total_stress, cpt.depth, label='Total Stress', color='black')
plt.plot(cpt.hydro_pore_pressure, cpt.depth, label='Pore Pressure', color='darkblue')
plt.plot(cpt.pore_pressure_u2 * 1000, cpt.depth, label='Pore Pressure U2', alpha=0.4, color='grey')
plt.xlabel('Stress (kPa)')
plt.ylabel('Depth NAP (m)')
plt.title(f'Total and effective stress - CPT{cpt.name}')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()  # Invert y-axis to have depth increasing downwards
# limit x from 0 to 200
plt.xlim(-5, 200)

plt.show()

























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
#
#
#
#
#
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
