import os
import numpy as np
import matplotlib.pyplot as plt

def read_ocv_file(filepath):
    """
    Lit un fichier .ocv :
    Colonne 0 = longueur d'onde
    Colonne 1 = absorption
    """
    data = np.loadtxt(filepath)
    wavelength = data[:, 0]
    absorption = data[:, 1]
    return wavelength, absorption


def load_folder(folder_path):
    """
    Charge tous les fichiers .ocv d'un dossier
    et retourne la moyenne d'absorption
    """
    absorptions = []
    wavelength_ref = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".ocv"):
            filepath = os.path.join(folder_path, filename)
            wavelength, absorption = read_ocv_file(filepath)

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

# Chargement des données
w1, a1 = load_folder(folder1)
w2, a2 = load_folder(folder2)
w3, a3 = load_folder(folder3)

# === GRAPHIQUE ===
plt.figure()

plt.plot(w1, a1, label="Dossier 1")
plt.plot(w2, a2, label="Dossier 2")
plt.plot(w3, a3, label="Dossier 3")

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorption")
plt.title("Absorption en fonction de la longueur d'onde")

plt.legend()
plt.grid(True)

plt.show()
