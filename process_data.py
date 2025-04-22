import pickle
import pandas as pd
import matplotlib.pyplot as plt

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod, ShearWaveVelocityMethod, OCRMethod


# TO LOAD THE PICKLE FILE FOR BAVOIS
bavois_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT\cpt_bavois.pkl"
with open(bavois_pickle, "rb") as f:
    cpt_bavois = pickle.load(f)

# TO LOAD THE PICKLE FILE FOR CHAVORNAY
chavornay_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT\cpt_chavornay.pkl"
with open(chavornay_pickle, "rb") as f:
    cpt_chavornay = pickle.load(f)


#random_cpt_path = r"C:\Users\camposmo\Stichting Deltares\RESET - Documents\Activities year 4\2a - Strength modeling\data\cpt\CPT000000217362_IMBRO.gef"
# cpt = GefCpt()
# cpt.read(random_cpt_path)


# Load the CSV with additional CPT information
metadata_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\cpt_other_details.csv"
metadata_df = pd.read_csv(metadata_path)

print(metadata_df.columns)


def create_cpt_object(cpt_name: str, cpt_dict: dict, metadata_df: pd.DataFrame, site_name: str) -> GefCpt:
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
        cpt.local_reference_level = -row["elev_cpt"].item()
    else:
        cpt.name = cpt_name
        cpt.coordinates = [None, None]
        cpt.local_reference_level = None
        cpt.a = None

    return cpt


def process_cpt_dict(cpt_dict: dict, metadata_df: pd.DataFrame, site_name: str) -> list:
    cpt_objects = []
    for cpt_name in cpt_dict:
        try:
            cpt_obj = create_cpt_object(cpt_name, cpt_dict, metadata_df, site_name)
            cpt_objects.append(cpt_obj)
        except Exception as e:
            print(f"[WARNING] Skipped {cpt_name} due to error: {e}")
    return cpt_objects


cpt_bavois_list = process_cpt_dict(cpt_bavois, metadata_df, site_name="bavois")
cpt_chavornay_list = process_cpt_dict(cpt_chavornay, metadata_df, site_name="chavornay")


print(cpt_bavois_list)


























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


print(cpt)











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
