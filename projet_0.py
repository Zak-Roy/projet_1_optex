import os
import numpy as np
import matplotlib.pyplot as plt

def read_txt_file(filepath):
    """
    Lit un fichier .txt contenant :
    colonne 0 = longueur d'onde
    colonne 1 = absorption
    """
    try:
        data = np.loadtxt(filepath)
    except ValueError:
        # Si séparateur virgule
        data = np.loadtxt(filepath, delimiter=",")
    
    wavelength = data[:, 0]
    absorption = data[:, 1]
    
    return wavelength, absorption


def load_folder(folder_path):
    """
    Charge tous les fichiers .txt d'un dossier
    et retourne la moyenne des absorptions
    """
    absorptions = []
    wavelength_ref = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            wavelength, absorption = read_txt_file(filepath)

            if wavelength_ref is None:
                wavelength_ref = wavelength

            absorptions.append(absorption)

    absorptions = np.array(absorptions)
    absorption_mean = np.mean(absorptions, axis=0)

    return wavelength_ref, absorption_mean


# === CHEMINS DES DOSSIERS ===
folder1 = "chemin/vers/dossier1"
folder2 = "chemin/vers/dossier2"
folder3 = "chemin/vers/dossier3"

folders = [folder1, folder2, folder3]

plt.figure()

for i, folder in enumerate(folders):
    wavelength, absorption = load_folder(folder)
    plt.plot(wavelength, absorption, label=f"Dossier {i+1}")

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorption")
plt.title("Absorption en fonction de la longueur d'onde")

plt.legend()
plt.grid(True)

plt.show()
