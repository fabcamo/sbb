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

random_cpt_path = r"C:\Users\camposmo\Stichting Deltares\RESET - Documents\Activities year 4\2a - Strength modeling\data\cpt\CPT000000217362_IMBRO.gef"

# Load the CSV with additional CPT information
metadata_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\cpt_other_details.csv"
metadata_df = pd.read_csv(metadata_path)

print(metadata_df.columns)



# Create manually the objects for the CPT CPTU01 in bavois
cpt = GefCpt()


cpt.tip = cpt_bavois['CPTU01']['qc']
cpt.penetration_length = cpt_bavois['CPTU01']['z']

# Get the row that contains the information for CPTU01 and bavois
row = metadata_df.query("site == 'bavois' and name == 'CPTU01'")
cpt.local_reference_level = row["elev_cpt"].squeeze() if not row.empty else None


cpt.pre_process_data()



#
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
