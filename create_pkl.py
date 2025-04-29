import os
import re
import pickle
import pandas as pd
import numpy as np


def read_cpt_txt(filepath, encoding='latin1'):
    """
    Reads a Bavois CPT .txt file with possible blank lines in metadata or data block.
    Preserves original column names and returns clean numeric data.
    """
    column_names = [
        "z", "qc", "fs", "u2", "|Rf|", "Bq", "su CSS",
        "St", "?'", "c'", "Id", "?'p", "ME1", "no_id"
    ]

    with open(filepath, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove blank lines

    # Detect header line (must start with "z")
    for i, line in enumerate(lines):
        if line.startswith("z"):
            header_idx = i
            break
    else:
        raise ValueError("Header line with 'z' not found.")

    # Skip the unit line (we assume it comes right after the header)
    data_start_idx = header_idx + 2

    # Parse valid data rows only
    data_rows = []
    for line in lines[data_start_idx:]:
        fields = re.split(r'\s+', line)
        if len(fields) == 14:
            data_rows.append(fields)

    # Build DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace(9999.00, np.nan, inplace=True)


    return df


def load_all_cpts(folder_path):
    cpt_data = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith('.txt'):
            cpt_name = os.path.splitext(file)[0]
            filepath = os.path.join(folder_path, file)
            try:
                df = read_cpt_txt(filepath)  # your existing function
                cpt_data[cpt_name] = {col: df[col].values for col in df.columns}
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return cpt_data


def save_cpt_data(data_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

# Path's
bavois_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT"
chavornay_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT"
ependes_path = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_33000-37500_Ependes-Yverdon site investigation\CPT Roh-Daten"

# Create the pickle files
cpt_bavois = load_all_cpts(bavois_path)
cpt_chavornay = load_all_cpts(chavornay_path)
cpt_ependes = load_all_cpts(ependes_path)

# Save the data to pickle files
save_cpt_data(cpt_bavois, os.path.join(bavois_path, 'cpt_bavois.pkl'))
save_cpt_data(cpt_chavornay, os.path.join(chavornay_path, 'cpt_chavornay.pkl'))
save_cpt_data(cpt_ependes, os.path.join(ependes_path, 'cpt_ependes.pkl'))