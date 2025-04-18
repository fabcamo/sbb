import pickle
import matplotlib.pyplot as plt


# TO LOAD THE PICKLE FILE FOR BAVOIS
bavois_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_22400-26400_Bavois site investigation\CPT\cpt_bavois.pkl"
with open(bavois_pickle, "rb") as f:
    cpt_bavois = pickle.load(f)

# TO LOAD THE PICKLE FILE FOR CHAVORNAY
chavornay_pickle = r"c:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\Geotechnical site investigations\L210_28200-33000_Chavornay-Ependes site investigation\CPT\cpt_chavornay.pkl"
with open(chavornay_pickle, "rb") as f:
    cpt_chavornay = pickle.load(f)



# Loop over each CPT and generate plots
for cpt_name, data in cpt_bavois.items():
    z = data.get('z')
    qc = data.get('qc')
    fs = data.get('fs')

    fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    # Plot qc
    axs[0].plot(qc, z, color='tab:blue')
    axs[0].set_title(f"{cpt_name} - qc")
    axs[0].set_xlabel("qc [MN/m²]")
    axs[0].set_ylabel("Depth z [m]")
    axs[0].grid(True)
    axs[0].set_ylim([max(z), min(z)])  # Invert y-axis

    # Plot fs
    axs[1].plot(fs, z, color='tab:red', linestyle='--')
    axs[1].set_title(f"{cpt_name} - fs")
    axs[1].set_xlabel("fs [MN/m²]")
    axs[1].grid(True)
    axs[1].set_ylim([max(z), min(z)])  # Invert y-axis

    plt.tight_layout()

    plt.show()
    plt.close()
