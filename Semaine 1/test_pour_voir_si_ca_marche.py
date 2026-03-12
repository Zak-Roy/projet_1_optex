import os
import numpy as np
import matplotlib.pyplot as plt

def read_txt_file(filepath):
    """
    Lit un fichier .txt contenant au minimum 2 colonnes :
    colonne 0 = longueur d'onde
    colonne 1 = absorption

    Gère : séparateurs espace/tab/virgule ; lignes d'en-tête/commentaires.
    """
    for kwargs in (dict(), dict(delimiter=",")):
        try:
            data = np.loadtxt(filepath, comments="#", **kwargs)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 2:
                raise ValueError("Moins de 2 colonnes détectées.")
            return data[:, 0], data[:, 1]
        except Exception:
            pass

    for skip in range(1, 50):
        for kwargs in (dict(skiprows=skip), dict(delimiter=",", skiprows=skip)):
            try:
                data = np.loadtxt(filepath, comments="#", **kwargs)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] < 2:
                    continue
                return data[:, 0], data[:, 1]
            except Exception:
                pass

    raise ValueError(f"Impossible de lire des données numériques (2 colonnes) dans : {filepath}")


def get_txt_files(folder_path):
    """Liste triée des .txt dans folder_path (non cachés)."""
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".txt") and not f.startswith(".")
    ]
    files.sort()
    return files


def plot_all(txt_files, title, x_max=None):
    """
    Trace toutes les courbes.
    Si x_max est donné, coupe à x <= x_max.
    """
    plt.figure()

    for filepath in txt_files:
        wavelength, absorption = read_txt_file(filepath)
        label = os.path.splitext(os.path.basename(filepath))[0]

        if x_max is not None:
            mask = wavelength <= x_max
            wavelength = wavelength[mask]
            absorption = absorption[mask]

        plt.plot(wavelength, absorption, label=label)

    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Absorption")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# === DOSSIER DU SCRIPT ===
base_dir = os.path.dirname(os.path.abspath(__file__))
txt_files = get_txt_files(base_dir)

if len(txt_files) == 0:
    raise FileNotFoundError(f"Aucun fichier .txt trouvé dans : {base_dir}")

print("Fichiers détectés :")
for f in txt_files:
    print(" -", os.path.basename(f))

# Graphique 1 : complet
plot_all(txt_files, "Absorption en fonction de la longueur d'onde (complet)")

# Graphique 2 : coupé à 1400 nm
plot_all(txt_files, "Absorption en fonction de la longueur d'onde (≤ 1400 nm)", x_max=1400)

# Affiche les 2 figures
plt.show()
