import pickle
import pandas as pd

from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod, ShearWaveVelocityMethod, OCRMethod

from calculations import calculate_vs_with_different_methods
from process_data import initialize_cpt_objects, save_results_as_csv, add_measured_vs_data

########################################################################################################################
# 1. Define the relevant paths
########################################################################################################################
# Pickled files with the CPT data
bavois_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT\cpt_bavois.pkl"
chavornay_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT\cpt_chavornay.pkl"
ependes_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_33000-37500_Ependes-Yverdon site investigation\CPT Roh-Daten\cpt_ependes.pkl"

# Metadata path
metadata_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\cpt_metadata.csv"

#  Paths for the pickel files containing the Lengkeek (2024) classification
L24R10_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\polygons_L24R10.pkl"

# Shear wave measurements in the SCPT path
SCPTu_bavois_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT\SDMT Report.xlsx"
SCPTu_chavornay_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT\SCPTU Werte.xlsx"
SCPTu_ependes_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_33000-37500_Ependes-Yverdon site investigation\CPT Roh-Daten\SCPTu VS summary.xlsx"

# Save results path
bavois_results_path = r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\bavois'
chavornay_results_path = r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\chavornay'
ependes_results_path = r'c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\results\ependes'

########################################################################################################################
# 2. Load the data
#########################################################################################################################

# Load the pickle files with the CPT data
with open(bavois_pickle, "rb") as f:
    cpt_bavois = pickle.load(f)

with open(chavornay_pickle, "rb") as f:
    cpt_chavornay = pickle.load(f)

with open(ependes_pickle, "rb") as f:
    cpt_ependes = pickle.load(f)

# Load the metadata for the CPTs (I manually built this)
metadata_df = pd.read_csv(metadata_path)

# Load the polygons from the pickle file for the L24R10 classification
with open(L24R10_path, "rb") as f:
    polygons_L24R10 = pickle.load(f)

# List of predefined colors for the zones
colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'black', 'grey']

########################################################################################################################
# 3. Process the data
#########################################################################################################################
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

# Process the Bavois CPTs
for cpt in cpt_bavois_list:
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    save_results_as_csv(cpt, cpt_bavois, bavois_results_path, vs_list, polygons_L24R10)

# Process the Chavornay CPTs
for cpt in cpt_chavornay_list:
    # pre-process the CPT
    cpt.pre_process_data()
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    save_results_as_csv(cpt, cpt_chavornay, chavornay_results_path, vs_list, polygons_L24R10)

# Process the Ependes CPTs
for cpt in cpt_ependes_list:
    # pre-process the CPT
    cpt.pre_process_data()
    # 1. Calculate Vs first (safe copies)
    vs_list = calculate_vs_with_different_methods(cpt, interpreter)
    # 2. Now FINAL interpretation
    cpt.interpret_cpt(interpreter)
    # 3. Now save
    save_results_as_csv(cpt, cpt_ependes, ependes_results_path, vs_list, polygons_L24R10)

# Add the measured shear wave velocity to the results
add_measured_vs_data(SCPTu_bavois_path, bavois_results_path)
add_measured_vs_data(SCPTu_chavornay_path, chavornay_results_path)
add_measured_vs_data(SCPTu_ependes_path, ependes_results_path)
