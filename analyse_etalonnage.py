import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = "results"

# Pour afficher les graphiques pendant l'exécution
SHOW_PLOTS = True

# Plage spectrale de ton spectromètre (proche IR)
WAVELENGTH_MIN = 950.0
WAVELENGTH_MAX = 1705.0

# Lissage léger
SMOOTH_WINDOW = 7

# Largeur de bande (nm) pour la recherche automatique
BAND_WIDTH_NM = 20.0

EPS = 1e-12


# ============================================================
# OUTILS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def moving_average(y, window=7):
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def safe_divide(a, b):
    return a / np.where(np.abs(b) < EPS, EPS, b)


def area_normalize(y):
    s = np.sum(y)
    if np.abs(s) < EPS:
        return np.zeros_like(y)
    return y / s


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < EPS:
        return np.nan
    return 1 - ss_res / ss_tot


def save_show_close(filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ============================================================
# LECTURE DES FICHIERS TXT (2 colonnes)
# ============================================================

def read_two_column_spectrum(filepath):
    """
    Lit un fichier texte avec deux colonnes :
        longueur_d_onde    intensité

    Exemple :
        953.279    19303.80469
        959.192    17137.01755

    Ignore les lignes non numériques.
    """
    wavelengths = []
    intensities = []

    float_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(float_pattern, line)

            if len(nums) >= 2:
                try:
                    wl = float(nums[0])
                    inten = float(nums[1])
                except ValueError:
                    continue

                if 900 <= wl <= 1800:
                    wavelengths.append(wl)
                    intensities.append(inten)

    if len(wavelengths) < 10:
        raise ValueError(f"Lecture impossible ou trop peu de points dans {filepath}")

    df = pd.DataFrame({
        "wavelength": np.array(wavelengths, dtype=float),
        "intensity": np.array(intensities, dtype=float)
    })

    df = (
        df.drop_duplicates(subset="wavelength")
          .sort_values("wavelength")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# RECONNAISSANCE DES FICHIERS
# ============================================================

def classify_file(filepath):
    name = Path(filepath).stem.lower()

    if "dark" in name or "noir" in name:
        return {"type": "dark", "concentration": None, "label": "dark"}

    if re.search(r"\beau[_\- ]?0\b", name):
        return {"type": "water", "concentration": 0.0, "label": "eau_0"}

    m = re.search(r"eau[_\- ]?sucre[_\- ]?(\d+(?:[.,]\d+)?)", name)
    if m:
        c = float(m.group(1).replace(",", "."))
        return {"type": "sugar", "concentration": c, "label": f"eau_sucre_{c:g}"}

    m2 = re.search(r"sucre[_\- ]?(\d+(?:[.,]\d+)?)", name)
    if m2:
        c = float(m2.group(1).replace(",", "."))
        return {"type": "sugar", "concentration": c, "label": f"eau_sucre_{c:g}"}

    return {"type": "unknown", "concentration": None, "label": name}


# ============================================================
# INTERPOLATION SUR UNE GRILLE COMMUNE
# ============================================================

def build_common_grid(all_dfs, wl_min, wl_max):
    mins = [df["wavelength"].min() for df in all_dfs]
    maxs = [df["wavelength"].max() for df in all_dfs]

    common_min = max(wl_min, max(mins))
    common_max = min(wl_max, min(maxs))

    if common_max <= common_min:
        raise ValueError("Pas de recouvrement spectral commun entre les fichiers.")

    base = all_dfs[0]["wavelength"].values
    grid = base[(base >= common_min) & (base <= common_max)]

    if len(grid) < 20:
        grid = np.linspace(common_min, common_max, 1000)

    return grid


def interp_to_grid(df, grid):
    return np.interp(grid, df["wavelength"].values, df["intensity"].values)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_all_spectra(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    if not files:
        raise FileNotFoundError(f"Aucun fichier .txt trouvé dans {data_dir}")

    entries = []
    for fp in files:
        try:
            df = read_two_column_spectrum(fp)
            meta = classify_file(fp)

            print(f"[OK] {fp} -> {len(df)} points lus")

            entries.append({
                "filepath": fp,
                "meta": meta,
                "df": df
            })
        except Exception as e:
            print(f"[WARN] Fichier ignoré : {fp} -> {e}")

    if not entries:
        raise RuntimeError("Aucun fichier valide n'a pu être lu.")

    return entries


# ============================================================
# MOYENNE DES FICHIERS D'UNE MÊME CONDITION
# ============================================================

def average_group(spectra_list):
    arr = np.vstack(spectra_list)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros(arr.shape[1])
    return mean, std


# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    entries = load_all_spectra(DATA_DIR)
    all_dfs = [e["df"] for e in entries]
    grid = build_common_grid(all_dfs, WAVELENGTH_MIN, WAVELENGTH_MAX)

    for e in entries:
        e["y_interp"] = interp_to_grid(e["df"], grid)

    dark_spectra = []
    water_spectra = []
    sugar_groups = {}
    unknown_files = []

    for e in entries:
        meta = e["meta"]
        y = e["y_interp"]

        if meta["type"] == "dark":
            dark_spectra.append(y)
        elif meta["type"] == "water":
            water_spectra.append(y)
        elif meta["type"] == "sugar":
            c = meta["concentration"]
            sugar_groups.setdefault(c, []).append(y)
        else:
            unknown_files.append(e["filepath"])

    if unknown_files:
        print("\nFichiers non reconnus :")
        for fp in unknown_files:
            print("  ", fp)

    if len(dark_spectra) == 0:
        raise RuntimeError("Il faut au moins un fichier dark.")
    if len(water_spectra) == 0:
        raise RuntimeError("Il faut au moins un fichier eau_0.")
    if len(sugar_groups) == 0:
        raise RuntimeError("Il faut au moins un fichier eau_sucre_X.")

    print("\n=== Conditions détectées ===")
    print(f"dark: {len(dark_spectra)} fichier(s)")
    print(f"eau_0: {len(water_spectra)} fichier(s)")
    for c in sorted(sugar_groups):
        print(f"eau_sucre_{c:g}: {len(sugar_groups[c])} fichier(s)")

    print("\n=== Diagnostic lecture ===")
    print(f"Nombre de points spectraux : {len(grid)}")
    print(f"Plage utilisée : {grid[0]:.3f} nm à {grid[-1]:.3f} nm")

    # Moyennes dark et eau
    dark_mean, dark_std = average_group(dark_spectra)
    water_mean_raw, water_std_raw = average_group(water_spectra)

    # Correction dark pour eau
    water_corr = np.clip(water_mean_raw - dark_mean, EPS, None)

    rawcorr_by_conc = {}
    transmission_by_conc = {}
    normalized_by_conc = {}
    summary_rows = []

    for c in sorted(sugar_groups):
        sample_mean_raw, sample_std_raw = average_group(sugar_groups[c])

        sample_corr = np.clip(sample_mean_raw - dark_mean, EPS, None)
        transmission = safe_divide(sample_corr, water_corr)
        transmission_norm = area_normalize(transmission)
        transmission_norm = moving_average(transmission_norm, SMOOTH_WINDOW)

        rawcorr_by_conc[c] = sample_corr
        transmission_by_conc[c] = transmission
        normalized_by_conc[c] = transmission_norm

        summary_rows.append({
            "concentration_percent": c,
            "n_files": len(sugar_groups[c]),
            "signal_total_after_dark": float(np.sum(sample_corr)),
            "mean_transmission": float(np.mean(transmission))
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("concentration_percent")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_conditions.csv"), index=False)

    # ========================================================
    # FIGURE 1 : spectres corrigés du dark
    # ========================================================
    plt.figure(figsize=(10, 6))
    for c in sorted(rawcorr_by_conc):
        plt.plot(grid, rawcorr_by_conc[c], label=f"{c:g}%")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Intensité corrigée du dark")
    plt.title("Spectres corrigés du dark")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "01_spectres_corriges_dark.png"))

    # ========================================================
    # FIGURE 2 : transmission relative à l'eau
    # ========================================================
    plt.figure(figsize=(10, 6))
    for c in sorted(transmission_by_conc):
        plt.plot(grid, transmission_by_conc[c], label=f"{c:g}%")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission relative à l'eau")
    plt.title("Transmission relative à l'eau")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "02_transmission_relative.png"))

    # ========================================================
    # FIGURE 3 : transmission normalisée
    # ========================================================
    plt.figure(figsize=(10, 6))
    for c in sorted(normalized_by_conc):
        plt.plot(grid, normalized_by_conc[c], label=f"{c:g}%")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission normalisée")
    plt.title("Transmission normalisée (forme seulement)")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "03_transmission_normalisee.png"))

    # ========================================================
    # RECHERCHE DES BANDES LES PLUS SENSIBLES
    # ========================================================
    concentrations = np.array(sorted(normalized_by_conc.keys()), dtype=float)
    Y = np.vstack([normalized_by_conc[c] for c in concentrations])

    corr = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        yi = Y[:, i]
        if np.std(yi) < EPS or np.std(concentrations) < EPS:
            corr[i] = 0.0
        else:
            corr[i] = np.corrcoef(concentrations, yi)[0, 1]

    wl_step = np.median(np.diff(grid))
    band_pts = max(3, int(round(BAND_WIDTH_NM / wl_step)))
    corr_smooth = moving_average(corr, band_pts)

    idx_pos = int(np.argmax(corr_smooth))
    idx_neg = int(np.argmin(corr_smooth))

    half = band_pts // 2
    pos_start = max(0, idx_pos - half)
    pos_end = min(len(grid), idx_pos + half + 1)

    neg_start = max(0, idx_neg - half)
    neg_end = min(len(grid), idx_neg + half + 1)

    band_pos = (grid[pos_start], grid[pos_end - 1])
    band_neg = (grid[neg_start], grid[neg_end - 1])

    # ========================================================
    # FIGURE 4 : corrélation avec la concentration
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(grid, corr, label="Corrélation brute")
    plt.plot(grid, corr_smooth, label="Corrélation lissée")
    plt.axvspan(band_pos[0], band_pos[1], alpha=0.2,
                label=f"Bande + : {band_pos[0]:.1f}-{band_pos[1]:.1f} nm")
    plt.axvspan(band_neg[0], band_neg[1], alpha=0.2,
                label=f"Bande - : {band_neg[0]:.1f}-{band_neg[1]:.1f} nm")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Corrélation avec la concentration")
    plt.title("Bandes spectrales les plus sensibles à la concentration")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "04_correlation_concentration.png"))

    # ========================================================
    # RATIO SPECTRAL ET COURBE D'ÉTALONNAGE
    # ========================================================
    band_pos_vals = np.mean(Y[:, pos_start:pos_end], axis=1)
    band_neg_vals = np.mean(Y[:, neg_start:neg_end], axis=1)
    ratio = safe_divide(band_pos_vals, band_neg_vals)

    coeffs = np.polyfit(ratio, concentrations, deg=1)
    a, b = coeffs[0], coeffs[1]
    pred = a * ratio + b
    r2 = r_squared(concentrations, pred)

    calib_df = pd.DataFrame({
        "concentration_percent": concentrations,
        "band_pos_mean": band_pos_vals,
        "band_neg_mean": band_neg_vals,
        "ratio_pos_over_neg": ratio,
        "predicted_concentration_percent": pred,
        "residual_percent": pred - concentrations
    })
    calib_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_table.csv"), index=False)

    # ========================================================
    # FIGURE 5 : courbe d'étalonnage
    # ========================================================
    xfit = np.linspace(np.min(ratio), np.max(ratio), 200)
    yfit = a * xfit + b

    plt.figure(figsize=(8, 6))
    plt.scatter(ratio, concentrations, label="Données")
    plt.plot(xfit, yfit, label=f"C = {a:.4g} R + {b:.4g}\nR² = {r2:.4f}")
    plt.xlabel("Ratio spectral")
    plt.ylabel("Concentration (%)")
    plt.title("Courbe d'étalonnage")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "05_courbe_etalonnage.png"))

    # ========================================================
    # EXPORT DES SPECTRES
    # ========================================================
    out_spec = pd.DataFrame({"wavelength_nm": grid})
    out_spec["dark_mean"] = dark_mean
    out_spec["water_mean_raw"] = water_mean_raw
    out_spec["water_corr"] = water_corr

    for c in sorted(rawcorr_by_conc):
        out_spec[f"corr_{c:g}pct"] = rawcorr_by_conc[c]
        out_spec[f"trans_{c:g}pct"] = transmission_by_conc[c]
        out_spec[f"norm_{c:g}pct"] = normalized_by_conc[c]

    out_spec.to_csv(os.path.join(OUTPUT_DIR, "spectres_moyens.csv"), index=False)

    # ========================================================
    # RAPPORT TEXTE
    # ========================================================
    with open(os.path.join(OUTPUT_DIR, "rapport_resume.txt"), "w", encoding="utf-8") as f:
        f.write("=== RÉSUMÉ DE L'ANALYSE ===\n\n")
        f.write("Méthode utilisée :\n")
        f.write("1) Lecture des fichiers 2 colonnes (lambda, intensité)\n")
        f.write("2) Soustraction du dark\n")
        f.write("3) Transmission relative à l'eau\n")
        f.write("4) Normalisation de forme\n")
        f.write("5) Recherche automatique des bandes sensibles\n")
        f.write("6) Ratio spectral et étalonnage\n\n")
        f.write(f"Plage analysée : {grid[0]:.3f} à {grid[-1]:.3f} nm\n")
        f.write(f"Bande + : {band_pos[0]:.2f} à {band_pos[1]:.2f} nm\n")
        f.write(f"Bande - : {band_neg[0]:.2f} à {band_neg[1]:.2f} nm\n")
        f.write(f"Calibration : C = {a:.6g} * R + {b:.6g}\n")
        f.write(f"R² = {r2:.6f}\n")

    print("\n=== Analyse terminée ===")
    print(f"Bande + : {band_pos[0]:.1f} à {band_pos[1]:.1f} nm")
    print(f"Bande - : {band_neg[0]:.1f} à {band_neg[1]:.1f} nm")
    print(f"Calibration : C = {a:.4g} * R + {b:.4g}")
    print(f"R² = {r2:.4f}")
    print(f"Résultats enregistrés dans : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()