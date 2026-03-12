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
OUTPUT_DIR = "results_boissons"

SHOW_PLOTS = True

WAVELENGTH_MIN = 950.0
WAVELENGTH_MAX = 1705.0

SMOOTH_WINDOW = 7
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
        return {"type": "standard", "concentration": c, "label": f"eau_sucre_{c:g}"}

    if "creme_soda_flat" in name:
        return {"type": "unknown", "concentration": None, "label": "creme_soda_flat"}

    if "sprite_flat" in name:
        return {"type": "unknown", "concentration": None, "label": "sprite_flat"}

    if "tonic_flat" in name:
        return {"type": "unknown", "concentration": None, "label": "tonic_flat"}

    return {"type": "unknown_other", "concentration": None, "label": name}


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
# CHARGEMENT
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

            print(f"[OK] {fp} -> {len(df)} points lus | type={meta['type']} | label={meta['label']}")

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


def average_group(spectra_list):
    arr = np.vstack(spectra_list)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros(arr.shape[1])
    return mean, std


# ============================================================
# FONCTIONS D'ANALYSE
# ============================================================

def compute_processed_spectrum(raw_mean, dark_mean, water_corr):
    sample_corr = np.clip(raw_mean - dark_mean, EPS, None)
    transmission = safe_divide(sample_corr, water_corr)
    transmission_norm = area_normalize(transmission)
    transmission_norm = moving_average(transmission_norm, SMOOTH_WINDOW)
    return sample_corr, transmission, transmission_norm


def estimate_concentration_from_ratio(ratio, a, b):
    return a * ratio + b


# ============================================================
# MAIN
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
    standard_groups = {}
    unknown_groups = {}
    ignored_files = []

    for e in entries:
        meta = e["meta"]
        y = e["y_interp"]

        if meta["type"] == "dark":
            dark_spectra.append(y)
        elif meta["type"] == "water":
            water_spectra.append(y)
        elif meta["type"] == "standard":
            c = meta["concentration"]
            standard_groups.setdefault(c, []).append(y)
        elif meta["type"] == "unknown":
            label = meta["label"]
            unknown_groups.setdefault(label, []).append(y)
        else:
            ignored_files.append(e["filepath"])

    if ignored_files:
        print("\nFichiers ignorés :")
        for fp in ignored_files:
            print("  ", fp)

    if len(dark_spectra) == 0:
        raise RuntimeError("Il faut au moins un fichier dark.")
    if len(water_spectra) == 0:
        raise RuntimeError("Il faut au moins un fichier eau_0.")
    if len(standard_groups) == 0:
        raise RuntimeError("Il faut des fichiers eau_sucre_X pour bâtir la calibration.")
    if len(unknown_groups) == 0:
        raise RuntimeError("Aucune boisson inconnue détectée (sprite_flat ou creme_soda_flat).")

    print("\n=== Résumé des groupes ===")
    print(f"dark: {len(dark_spectra)}")
    print(f"eau_0: {len(water_spectra)}")
    for c in sorted(standard_groups):
        print(f"standard {c:g}%: {len(standard_groups[c])}")
    for label in sorted(unknown_groups):
        print(f"inconnue {label}: {len(unknown_groups[label])}")

    print(f"\nPlage analysée : {grid[0]:.3f} à {grid[-1]:.3f} nm")
    print(f"Nombre de points : {len(grid)}")

    # Références
    dark_mean, _ = average_group(dark_spectra)
    water_mean_raw, _ = average_group(water_spectra)
    water_corr = np.clip(water_mean_raw - dark_mean, EPS, None)

    # --------------------------------------------------------
    # 1) Traiter les standards
    # --------------------------------------------------------
    std_rawcorr = {}
    std_trans = {}
    std_norm = {}

    for c in sorted(standard_groups):
        raw_mean, _ = average_group(standard_groups[c])
        sample_corr, transmission, transmission_norm = compute_processed_spectrum(
            raw_mean, dark_mean, water_corr
        )
        std_rawcorr[c] = sample_corr
        std_trans[c] = transmission
        std_norm[c] = transmission_norm

    # --------------------------------------------------------
    # 2) Trouver les bandes à partir des standards
    # --------------------------------------------------------
    concentrations = np.array(sorted(std_norm.keys()), dtype=float)
    Y_std = np.vstack([std_norm[c] for c in concentrations])

    corr = np.zeros(Y_std.shape[1])
    for i in range(Y_std.shape[1]):
        yi = Y_std[:, i]
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

    # --------------------------------------------------------
    # 3) Calibration
    # --------------------------------------------------------
    band_pos_vals = np.mean(Y_std[:, pos_start:pos_end], axis=1)
    band_neg_vals = np.mean(Y_std[:, neg_start:neg_end], axis=1)
    ratio_std = safe_divide(band_pos_vals, band_neg_vals)

    coeffs = np.polyfit(ratio_std, concentrations, deg=1)
    a, b = coeffs[0], coeffs[1]
    pred_std = a * ratio_std + b
    r2 = r_squared(concentrations, pred_std)

    calib_df = pd.DataFrame({
        "concentration_percent": concentrations,
        "band_pos_mean": band_pos_vals,
        "band_neg_mean": band_neg_vals,
        "ratio": ratio_std,
        "predicted_concentration_percent": pred_std,
        "residual_percent": pred_std - concentrations
    })
    calib_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_table.csv"), index=False)

    # --------------------------------------------------------
    # 4) Traiter les boissons inconnues
    # --------------------------------------------------------
    unknown_results = []

    unk_rawcorr = {}
    unk_trans = {}
    unk_norm = {}

    for label in sorted(unknown_groups):
        raw_mean, raw_std = average_group(unknown_groups[label])
        sample_corr, transmission, transmission_norm = compute_processed_spectrum(
            raw_mean, dark_mean, water_corr
        )

        unk_rawcorr[label] = sample_corr
        unk_trans[label] = transmission
        unk_norm[label] = transmission_norm

        band_pos_mean = float(np.mean(transmission_norm[pos_start:pos_end]))
        band_neg_mean = float(np.mean(transmission_norm[neg_start:neg_end]))
        ratio = float(safe_divide(np.array([band_pos_mean]), np.array([band_neg_mean]))[0])
        estimated_c = float(estimate_concentration_from_ratio(ratio, a, b))

        # position relative au nuage de calibration
        ratio_min = float(np.min(ratio_std))
        ratio_max = float(np.max(ratio_std))
        in_range = (ratio_min <= ratio <= ratio_max)

        unknown_results.append({
            "label": label,
            "band_pos_mean": band_pos_mean,
            "band_neg_mean": band_neg_mean,
            "ratio": ratio,
            "estimated_concentration_percent": estimated_c,
            "ratio_in_calibration_range": in_range
        })

    unknown_df = pd.DataFrame(unknown_results)
    unknown_df.to_csv(os.path.join(OUTPUT_DIR, "unknown_beverages_estimates.csv"), index=False)

    # --------------------------------------------------------
    # FIGURE 1 : Standards - transmission relative
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for c in sorted(std_trans):
        plt.plot(grid, std_trans[c], label=f"{c:g}%")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission relative à l'eau")
    plt.title("Solutions étalons : transmission relative à l'eau")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "01_standards_transmission_relative.png"))

    # --------------------------------------------------------
    # FIGURE 2 : Standards - transmission normalisée
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for c in sorted(std_norm):
        plt.plot(grid, std_norm[c], label=f"{c:g}%")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission normalisée")
    plt.title("Solutions étalons : transmission normalisée")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "02_standards_transmission_normalisee.png"))

    # --------------------------------------------------------
    # FIGURE 3 : Corrélation et bandes
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(grid, corr, label="Corrélation brute")
    plt.plot(grid, corr_smooth, label="Corrélation lissée")
    plt.axvspan(band_pos[0], band_pos[1], alpha=0.2,
                label=f"Bande + : {band_pos[0]:.1f}-{band_pos[1]:.1f} nm")
    plt.axvspan(band_neg[0], band_neg[1], alpha=0.2,
                label=f"Bande - : {band_neg[0]:.1f}-{band_neg[1]:.1f} nm")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Corrélation avec la concentration")
    plt.title("Bandes spectrales choisies à partir des étalons")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "03_correlation_et_bandes.png"))

    # --------------------------------------------------------
    # FIGURE 4 : Inconnues vs standards (transmission relative)
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for c in sorted(std_trans):
        plt.plot(grid, std_trans[c], alpha=0.35, linewidth=1)
    for label in sorted(unk_trans):
        plt.plot(grid, unk_trans[label], linewidth=2.5, label=label)
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission relative à l'eau")
    plt.title("Boissons inconnues comparées aux étalons")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "04_unknowns_vs_standards_transmission.png"))

    # --------------------------------------------------------
    # FIGURE 5 : Inconnues vs standards (normalisé)
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for c in sorted(std_norm):
        plt.plot(grid, std_norm[c], alpha=0.35, linewidth=1)
    for label in sorted(unk_norm):
        plt.plot(grid, unk_norm[label], linewidth=2.5, label=label)
    plt.axvspan(band_pos[0], band_pos[1], alpha=0.15)
    plt.axvspan(band_neg[0], band_neg[1], alpha=0.15)
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission normalisée")
    plt.title("Boissons inconnues sur les bandes utiles")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "05_unknowns_vs_standards_normalized.png"))

    # --------------------------------------------------------
    # FIGURE 6 : Courbe d'étalonnage + inconnues
    # --------------------------------------------------------
    xfit = np.linspace(np.min(ratio_std), np.max(ratio_std), 300)
    yfit = a * xfit + b

    plt.figure(figsize=(8, 6))
    plt.scatter(ratio_std, concentrations, label="Étalons")
    plt.plot(xfit, yfit, label=f"C = {a:.4g} R + {b:.4g}\nR² = {r2:.4f}")

    for row in unknown_results:
        plt.scatter(row["ratio"], row["estimated_concentration_percent"], s=90, marker="x", label=row["label"])

    plt.xlabel("Ratio spectral")
    plt.ylabel("Concentration (%)")
    plt.title("Courbe d'étalonnage avec boissons inconnues")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "06_calibration_with_unknowns.png"))

    # --------------------------------------------------------
    # FIGURE 7 : Zoom bandes utiles
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for c in sorted(std_norm):
        plt.plot(grid, std_norm[c], alpha=0.4, linewidth=1)
    for label in sorted(unk_norm):
        plt.plot(grid, unk_norm[label], linewidth=2.5, label=label)
    plt.xlim(min(band_neg[0], band_pos[0]) - 20, max(band_neg[1], band_pos[1]) + 20)
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Transmission normalisée")
    plt.title("Zoom sur les bandes utilisées pour l'estimation")
    plt.legend()
    save_show_close(os.path.join(OUTPUT_DIR, "07_zoom_bandes_utiles.png"))

    # --------------------------------------------------------
    # Export spectres
    # --------------------------------------------------------
    out_spec = pd.DataFrame({"wavelength_nm": grid})
    out_spec["dark_mean"] = dark_mean
    out_spec["water_mean_raw"] = water_mean_raw
    out_spec["water_corr"] = water_corr

    for c in sorted(std_trans):
        out_spec[f"std_trans_{c:g}pct"] = std_trans[c]
        out_spec[f"std_norm_{c:g}pct"] = std_norm[c]

    for label in sorted(unk_trans):
        out_spec[f"{label}_trans"] = unk_trans[label]
        out_spec[f"{label}_norm"] = unk_norm[label]

    out_spec.to_csv(os.path.join(OUTPUT_DIR, "all_processed_spectra.csv"), index=False)

    # --------------------------------------------------------
    # Rapport texte
    # --------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, "rapport_boissons.txt"), "w", encoding="utf-8") as f:
        f.write("=== ANALYSE DES BOISSONS INCONNUES ===\n\n")
        f.write("Méthode:\n")
        f.write("1) calibration sur les solutions eau + sucre\n")
        f.write("2) choix automatique de deux bandes sensibles\n")
        f.write("3) construction d'un ratio spectral R\n")
        f.write("4) estimation de la concentration par C = aR + b\n\n")

        f.write(f"Plage analysée : {grid[0]:.3f} à {grid[-1]:.3f} nm\n")
        f.write(f"Bande + : {band_pos[0]:.2f} à {band_pos[1]:.2f} nm\n")
        f.write(f"Bande - : {band_neg[0]:.2f} à {band_neg[1]:.2f} nm\n")
        f.write(f"Calibration : C = {a:.6g} * R + {b:.6g}\n")
        f.write(f"R² = {r2:.6f}\n\n")

        f.write("Estimations des boissons:\n")
        for row in unknown_results:
            f.write(f"- {row['label']}:\n")
            f.write(f"    ratio = {row['ratio']:.6f}\n")
            f.write(f"    concentration estimée = {row['estimated_concentration_percent']:.3f} %\n")
            f.write(f"    dans la plage de calibration = {row['ratio_in_calibration_range']}\n")

    # --------------------------------------------------------
    # Résumé console
    # --------------------------------------------------------
    print("\n=== Calibration ===")
    print(f"Bande + : {band_pos[0]:.1f} à {band_pos[1]:.1f} nm")
    print(f"Bande - : {band_neg[0]:.1f} à {band_neg[1]:.1f} nm")
    print(f"C = {a:.4g} * R + {b:.4g}")
    print(f"R² = {r2:.4f}")

    print("\n=== Estimations des boissons ===")
    for row in unknown_results:
        print(
            f"{row['label']} -> "
            f"ratio = {row['ratio']:.6f}, "
            f"C_est = {row['estimated_concentration_percent']:.3f} %, "
            f"dans_plage = {row['ratio_in_calibration_range']}"
        )

    print(f"\nRésultats enregistrés dans : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()