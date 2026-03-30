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

# --- Exclure 10% de la calibration, mais le montrer dans les figures + calibration
EXCLUDE_CONCENTRATIONS = {10.0}
PLOT_EXCLUDED_STANDARDS = True  # True: 10% apparaît en pointillé/pâle dans les spectres

# --- Sélection bandes (différence extrêmes)
USE_BAND_SEARCH_RANGES = True
BAND_NEG_SEARCH_RANGE = (1000.0, 1120.0)  # ~1060
BAND_POS_SEARCH_RANGE = (1240.0, 1360.0)  # ~1300

# --- Sortie "rapport"
SAVE_PNG = True
SAVE_PDF = True
DPI = 400
FIGSIZE_WIDE = (10, 6)
FIGSIZE_CAL = (8, 6)
FONT_SIZE = 12

# ============================================================
# VALEURS "FABRICANT" (étiquettes) + tolérance Health Canada/CFIA
# Tolérance utilisée ici : ±20% (conformité étiquetage)
# ============================================================

SERVING_ML_DEFAULT = 375.0
MFG_REL_TOL = 0.20  # 20% relatif sur les grammes de sucre

# label dans vos fichiers -> (sucre_g, volume_ml)
MANUFACTURER_LABELS = {
    "sprite_flat":     (40.0, SERVING_ML_DEFAULT),
    "creme_soda_flat": (47.0, SERVING_ML_DEFAULT),
    "tonic_flat":      (33.0, SERVING_ML_DEFAULT),
}


# ============================================================
# OUTILS
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def moving_average(y, window=7):
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a / np.where(np.abs(b) < EPS, EPS, b)

def area_normalize(y):
    y = np.asarray(y, dtype=float)
    s = float(np.sum(y))
    if abs(s) < EPS:
        return np.zeros_like(y)
    return y / s

def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < EPS:
        return np.nan
    return 1 - ss_res / ss_tot

def save_fig(basepath_no_ext: str):
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(basepath_no_ext + ".png", dpi=DPI)
    if SAVE_PDF:
        plt.savefig(basepath_no_ext + ".pdf")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def idx_range(grid, wl_min, wl_max):
    m = (grid >= wl_min) & (grid <= wl_max)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)

def band_from_index(n_points, center_idx, band_pts):
    half = band_pts // 2
    start = max(0, center_idx - half)
    end = min(n_points, center_idx + half + 1)
    return start, end

def is_excluded_conc(c: float) -> bool:
    for x in EXCLUDE_CONCENTRATIONS:
        if abs(c - float(x)) < 1e-9:
            return True
    return False

def scale01(y, y_min, y_max):
    y = np.asarray(y, dtype=float)
    den = (y_max - y_min)
    if abs(den) < EPS:
        return np.zeros_like(y)
    return (y - y_min) / den

def global_minmax(curves):
    vmin = np.inf
    vmax = -np.inf
    for y in curves:
        y = np.asarray(y, dtype=float)
        vmin = min(vmin, float(np.min(y)))
        vmax = max(vmax, float(np.max(y)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if abs(vmax - vmin) < EPS:
        return vmin, vmin + 1.0
    return vmin, vmax

def prettify_axes(ax):
    ax.grid(False)  # PAS de quadrillage
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def manufacturer_percent(sugar_g, vol_ml, rel_tol=MFG_REL_TOL):
    """
    Convertit sucre (g / vol_ml) -> % (≈ g/100 mL)
    avec incertitude relative ±rel_tol sur sugar_g.
    """
    c = (sugar_g / vol_ml) * 100.0
    sugar_g_unc = rel_tol * sugar_g
    dc = (sugar_g_unc / vol_ml) * 100.0
    return float(c), float(dc)


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
            print(f"[OK] {fp} -> {len(df)} points | type={meta['type']} | label={meta['label']}")
            entries.append({"filepath": fp, "meta": meta, "df": df})
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
# PIPELINE (simple)
# ============================================================

def compute_T_smooth_from_raw(raw_y, dark_mean, water_corr):
    sample_corr = np.clip(raw_y - dark_mean, EPS, None)
    T_raw = safe_divide(sample_corr, water_corr)  # peut dépasser 1
    return moving_average(T_raw, SMOOTH_WINDOW)

def compute_S_for_calibration(T_smooth):
    # méthode legacy (normalisation par aire)
    return moving_average(area_normalize(T_smooth), SMOOTH_WINDOW)

def process_group(reps_rawy, dark_mean, water_corr):
    T_list = []
    S_list = []
    for y in reps_rawy:
        T_s = compute_T_smooth_from_raw(y, dark_mean, water_corr)
        S = compute_S_for_calibration(T_s)
        T_list.append(T_s)
        S_list.append(S)

    T_reps = np.vstack(T_list)
    S_reps = np.vstack(S_list)

    return {
        "T_reps": T_reps,
        "T_mean": np.mean(T_reps, axis=0),
        "T_std": np.std(T_reps, axis=0, ddof=1) if T_reps.shape[0] >= 2 else np.zeros(T_reps.shape[1]),
        "S_reps": S_reps,
        "S_mean": np.mean(S_reps, axis=0),
        "S_std": np.std(S_reps, axis=0, ddof=1) if S_reps.shape[0] >= 2 else np.zeros(S_reps.shape[1]),
        "n_rep": int(T_reps.shape[0]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.grid": False,
        "legend.frameon": True,
    })

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
            standard_groups.setdefault(meta["concentration"], []).append(y)
        elif meta["type"] == "unknown":
            unknown_groups.setdefault(meta["label"], []).append(y)
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
        raise RuntimeError("Aucune boisson inconnue détectée (sprite_flat / creme_soda_flat / tonic_flat).")

    # Références
    dark_mean, _ = average_group(dark_spectra)
    water_mean_raw, _ = average_group(water_spectra)
    water_corr = np.clip(water_mean_raw - dark_mean, EPS, None)

    # Process standards + inconnues
    std_proc_all = {c: process_group(standard_groups[c], dark_mean, water_corr) for c in sorted(standard_groups)}
    unk_proc = {lab: process_group(unknown_groups[lab], dark_mean, water_corr) for lab in sorted(unknown_groups)}

    # Standards utilisés pour calibration (10% exclu)
    std_used = {c: std_proc_all[c] for c in std_proc_all if not is_excluded_conc(c)}
    concentrations = np.array(sorted(std_used.keys()), dtype=float)
    if len(concentrations) < 3:
        raise RuntimeError("Après exclusion, il faut au moins 3 concentrations pour calibrer.")

    # ========================================================
    # 1) Bandes via ΔS entre extrêmes
    # ========================================================
    wl_step = np.median(np.diff(grid))
    band_pts = max(3, int(round(BAND_WIDTH_NM / wl_step)))

    cmin = float(np.min(concentrations))
    cmax = float(np.max(concentrations))
    Smin = std_used[cmin]["S_mean"]
    Smax = std_used[cmax]["S_mean"]

    delta = Smax - Smin
    delta_s = moving_average(delta, band_pts)

    if USE_BAND_SEARCH_RANGES:
        neg_rng = idx_range(grid, BAND_NEG_SEARCH_RANGE[0], BAND_NEG_SEARCH_RANGE[1])
        pos_rng = idx_range(grid, BAND_POS_SEARCH_RANGE[0], BAND_POS_SEARCH_RANGE[1])
        if (neg_rng is None) or (pos_rng is None):
            raise RuntimeError("Plages de recherche bandes invalides: hors du spectre mesuré.")
        idx_neg = int(np.argmin(delta_s[neg_rng[0]:neg_rng[1]]) + neg_rng[0])
        idx_pos = int(np.argmax(delta_s[pos_rng[0]:pos_rng[1]]) + pos_rng[0])
    else:
        idx_neg = int(np.argmin(delta_s))
        idx_pos = int(np.argmax(delta_s))

    pos_start, pos_end = band_from_index(len(grid), idx_pos, band_pts)
    neg_start, neg_end = band_from_index(len(grid), idx_neg, band_pts)
    band_pos = (grid[pos_start], grid[pos_end - 1])
    band_neg = (grid[neg_start], grid[neg_end - 1])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(grid, delta, label=f"Δ(λ)=S({cmax:.0f}%) − S({cmin:.0f}%)")
    ax.plot(grid, delta_s, label="Δ lissée")
    ax.axvspan(band_neg[0], band_neg[1], alpha=0.18, label=f"Bande − : {band_neg[0]:.0f}-{band_neg[1]:.0f} nm")
    ax.axvspan(band_pos[0], band_pos[1], alpha=0.18, label=f"Bande + : {band_pos[0]:.0f}-{band_pos[1]:.0f} nm")
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("ΔS(λ)")
    ax.set_title("Sélection des bandes")
    ax.legend()
    prettify_axes(ax)
    save_fig(os.path.join(OUTPUT_DIR, "01_selection_bandes"))

    # ========================================================
    # 2) Calibration sur R (10% exclu)
    # ========================================================
    ratio_mean = []
    ratio_std = []

    for c in concentrations:
        Sreps = std_used[c]["S_reps"]
        pos_vals = np.mean(Sreps[:, pos_start:pos_end], axis=1)
        neg_vals = np.mean(Sreps[:, neg_start:neg_end], axis=1)
        Rreps = safe_divide(pos_vals, neg_vals)

        ratio_mean.append(float(np.mean(Rreps)))
        ratio_std.append(float(np.std(Rreps, ddof=1) if len(Rreps) >= 2 else 0.0))

    ratio_mean = np.array(ratio_mean, dtype=float)
    ratio_std = np.array(ratio_std, dtype=float)

    a, b = np.polyfit(ratio_mean, concentrations, deg=1)
    pred = a * ratio_mean + b
    r2 = r_squared(concentrations, pred)

    pd.DataFrame({
        "concentration_percent": concentrations,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "predicted_concentration_percent": pred,
        "residual_percent": pred - concentrations
    }).to_csv(os.path.join(OUTPUT_DIR, "calibration_table_excl10.csv"), index=False)

    # Ratio du 10% (exclu) pour l'afficher sur la courbe finale
    R10_mean = None
    R10_std = None
    if 10.0 in std_proc_all:
        Sreps_10 = std_proc_all[10.0]["S_reps"]
        pos_10 = np.mean(Sreps_10[:, pos_start:pos_end], axis=1)
        neg_10 = np.mean(Sreps_10[:, neg_start:neg_end], axis=1)
        R10_reps = safe_divide(pos_10, neg_10)
        R10_mean = float(np.mean(R10_reps))
        R10_std = float(np.std(R10_reps, ddof=1) if len(R10_reps) >= 2 else 0.0)

    # ========================================================
    # 3) Estimer les inconnues (nos mesures)
    # ========================================================
    unknown_results = []
    rmin, rmax = float(np.min(ratio_mean)), float(np.max(ratio_mean))

    for label in sorted(unk_proc):
        Sreps = unk_proc[label]["S_reps"]
        pos_vals = np.mean(Sreps[:, pos_start:pos_end], axis=1)
        neg_vals = np.mean(Sreps[:, neg_start:neg_end], axis=1)
        Rreps = safe_divide(pos_vals, neg_vals)

        Rm = float(np.mean(Rreps))
        Rs = float(np.std(Rreps, ddof=1) if len(Rreps) >= 2 else 0.0)

        C_est = float(a * Rm + b)
        C_sigma = float(abs(a) * Rs)

        unknown_results.append({
            "label": label,
            "ratio_mean": Rm,
            "ratio_std": Rs,
            "estimated_concentration_percent": C_est,
            "estimated_concentration_sigma_percent": C_sigma,
            "ratio_in_calibration_range": (rmin <= Rm <= rmax)
        })

    unknown_df = pd.DataFrame(unknown_results)
    unknown_df.to_csv(os.path.join(OUTPUT_DIR, "unknown_beverages_estimates_excl10.csv"), index=False)

    # ========================================================
    # 4) Spectres en 0–1
    # ========================================================

    # --- (A) Standards T(λ) -> 0–1
    curves_T_std = []
    for c in sorted(std_proc_all):
        if is_excluded_conc(c) and (not PLOT_EXCLUDED_STANDARDS):
            continue
        curves_T_std.append(std_proc_all[c]["T_mean"])
    T_min, T_max = global_minmax(curves_T_std)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for c in sorted(std_proc_all):
        is_excl = is_excluded_conc(c)
        if is_excl and (not PLOT_EXCLUDED_STANDARDS):
            continue

        Tm = std_proc_all[c]["T_mean"]
        Ts = std_proc_all[c]["T_std"]

        T01 = scale01(Tm, T_min, T_max)
        T01_lo = scale01(Tm - Ts, T_min, T_max)
        T01_hi = scale01(Tm + Ts, T_min, T_max)

        alpha = 0.35 if is_excl else 1.0
        lw = 1.0 if is_excl else 1.7
        ls = "--" if is_excl else "-"
        label = f"{c:g}% (exclu)" if is_excl else f"{c:g}%"

        ax.plot(grid, np.clip(T01, 0, 1), alpha=alpha, linewidth=lw, linestyle=ls, label=label)

        if std_proc_all[c]["n_rep"] >= 2 and (not is_excl):
            ax.fill_between(grid, np.clip(T01_lo, 0, 1), np.clip(T01_hi, 0, 1), alpha=0.12)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("Intensité relative (0–1)")
    ax.set_title("Solutions étalons")
    ax.legend(ncol=2)
    prettify_axes(ax)
    save_fig(os.path.join(OUTPUT_DIR, "02_standards_T"))

    # --- (B) Inconnues vs étalons T(λ) -> 0–1
    curves_T_mix = []
    for c in sorted(std_proc_all):
        if is_excluded_conc(c) and (not PLOT_EXCLUDED_STANDARDS):
            continue
        curves_T_mix.append(std_proc_all[c]["T_mean"])
    for lab in sorted(unk_proc):
        curves_T_mix.append(unk_proc[lab]["T_mean"])
    Tm_min, Tm_max = global_minmax(curves_T_mix)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for c in sorted(std_proc_all):
        is_excl = is_excluded_conc(c)
        if is_excl and (not PLOT_EXCLUDED_STANDARDS):
            continue
        T01 = scale01(std_proc_all[c]["T_mean"], Tm_min, Tm_max)
        ax.plot(grid, np.clip(T01, 0, 1), alpha=(0.10 if is_excl else 0.20), linewidth=1)

    for lab in sorted(unk_proc):
        Tm = unk_proc[lab]["T_mean"]
        Ts = unk_proc[lab]["T_std"]
        T01 = scale01(Tm, Tm_min, Tm_max)
        ax.plot(grid, np.clip(T01, 0, 1), linewidth=2.4, label=lab)
        if unk_proc[lab]["n_rep"] >= 2:
            ax.fill_between(
                grid,
                np.clip(scale01(Tm - Ts, Tm_min, Tm_max), 0, 1),
                np.clip(scale01(Tm + Ts, Tm_min, Tm_max), 0, 1),
                alpha=0.12
            )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("Intensité relative (0–1)")
    ax.set_title("Boissons vs étalons")
    ax.legend()
    prettify_axes(ax)
    save_fig(os.path.join(OUTPUT_DIR, "03_unknowns_vs_standards_T"))

    # --- (C) ✅ GRAPHE REQUIS : Signal S utilisé pour le ratio + bandes (0–1)
    curves_S = []
    for c in sorted(std_proc_all):
        if is_excluded_conc(c) and (not PLOT_EXCLUDED_STANDARDS):
            continue
        curves_S.append(std_proc_all[c]["S_mean"])
    S_min, S_max = global_minmax(curves_S)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for c in sorted(std_proc_all):
        is_excl = is_excluded_conc(c)
        if is_excl and (not PLOT_EXCLUDED_STANDARDS):
            continue

        Sm = std_proc_all[c]["S_mean"]
        S01 = scale01(Sm, S_min, S_max)

        alpha = 0.35 if is_excl else 1.0
        lw = 1.0 if is_excl else 1.6
        ls = "--" if is_excl else "-"
        label = f"{c:g}% (exclu)" if is_excl else f"{c:g}%"

        ax.plot(grid, np.clip(S01, 0, 1), alpha=alpha, linewidth=lw, linestyle=ls, label=label)

    ax.axvspan(band_neg[0], band_neg[1], alpha=0.12, label="Bande −")
    ax.axvspan(band_pos[0], band_pos[1], alpha=0.12, label="Bande +")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("Intensité relative (0–1)")
    ax.set_title("Signal utilisé pour le ratio + bandes")
    ax.legend(ncol=2, loc="upper right")
    prettify_axes(ax)
    save_fig(os.path.join(OUTPUT_DIR, "04_signal_ratio_bandes"))

    # ========================================================
    # 5) Courbe d’étalonnage et validation (exp vs fabricant) + point 10% exclu
    # ========================================================

    DISPLAY_NAME = {
        "sprite_flat": "Sprite",
        "creme_soda_flat": "Crème soda",
        "tonic_flat": "Tonic",
    }

    # Couleurs cohérentes (même couleur pour exp et fabricant d'une même boisson)
    color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {}

    def get_color(key):
        if key not in colors:
            colors[key] = next(color_cycle)
        return colors[key]

    # Courbe de calibration
    xfit = np.linspace(float(np.min(ratio_mean)), float(np.max(ratio_mean)), 300)
    yfit = a * xfit + b

    # Petit décalage horizontal pour éviter superposition exp/fabricant
    x_span = float(np.max(ratio_mean) - np.min(ratio_mean))
    x_off = 0.01 * x_span if x_span > EPS else 0.001

    fig, ax = plt.subplots(figsize=FIGSIZE_CAL)

    # Étalons (pas de "(fit)")
    ax.errorbar(ratio_mean, concentrations, xerr=ratio_std, fmt="o", capsize=3, label="Étalons")
    ax.plot(xfit, yfit, label=f"C = {a:.3g} R + {b:.3g}   |   R² = {r2:.3f}")

    # Point 10% exclu : étoile, sans légende
    if R10_mean is not None:
        ax.plot(R10_mean, 10.0, marker="*", markersize=14, linestyle="None", color="tab:blue")
        ax.annotate("10% exclu", (R10_mean, 10.0), textcoords="offset points", xytext=(6, 6))

    # Boissons : exp (plein) + fabricant (vide) + incertitudes
    for row in unknown_results:
        file_label = row["label"]
        name = DISPLAY_NAME.get(file_label, file_label)
        col = get_color(file_label)

        # Expérimental (plein)
        ax.errorbar(
            row["ratio_mean"],
            row["estimated_concentration_percent"],
            xerr=row["ratio_std"] if row["ratio_std"] > 0 else None,
            yerr=row["estimated_concentration_sigma_percent"] if row["ratio_std"] > 0 else None,
            fmt="o",
            markersize=10,
            capsize=3,
            markerfacecolor=col,
            markeredgecolor=col,
            linestyle="None",
            label=name
        )

        # Fabricant (vide) + tolérance Health Canada/CFIA (±20% sur g)
        if file_label in MANUFACTURER_LABELS:
            sugar_g, vol_ml = MANUFACTURER_LABELS[file_label]
            c_true, c_true_unc = manufacturer_percent(sugar_g, vol_ml, rel_tol=MFG_REL_TOL)

            ax.errorbar(
                row["ratio_mean"] + x_off,
                c_true,
                yerr=c_true_unc,
                fmt="o",
                markersize=10,
                capsize=3,
                markerfacecolor="none",
                markeredgecolor=col,
                markeredgewidth=1.8,
                linestyle="None",
                label="_nolegend_"
            )

    ax.set_xlabel("Ratio spectral R")
    ax.set_ylabel("Concentration sucre (%)")
    ax.set_title("Courbe d’étalonnage et validation")
    ax.legend(loc="upper left", framealpha=0)
    prettify_axes(ax)
    save_fig(os.path.join(OUTPUT_DIR, "05_calibration_validation"))

    # ========================================================
    # EXPORTS
    # ========================================================
    out_spec = pd.DataFrame({"wavelength_nm": grid})
    out_spec["dark_mean"] = dark_mean
    out_spec["water_mean_raw"] = water_mean_raw
    out_spec["water_corr"] = water_corr

    for c in sorted(std_proc_all):
        out_spec[f"std_T_mean_{c:g}pct"] = std_proc_all[c]["T_mean"]
        out_spec[f"std_T_std_{c:g}pct"] = std_proc_all[c]["T_std"]
        out_spec[f"std_S_mean_{c:g}pct"] = std_proc_all[c]["S_mean"]

    for lab in sorted(unk_proc):
        out_spec[f"{lab}_T_mean"] = unk_proc[lab]["T_mean"]
        out_spec[f"{lab}_T_std"] = unk_proc[lab]["T_std"]
        out_spec[f"{lab}_S_mean"] = unk_proc[lab]["S_mean"]

    out_spec.to_csv(os.path.join(OUTPUT_DIR, "all_processed_spectra.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "rapport_boissons.txt"), "w", encoding="utf-8") as f:
        f.write("=== Résultats ===\n\n")
        f.write("T(λ) = (I - Idark)/(I0 - Idark)\n")
        f.write("S(λ) = normalisation par aire de T(λ) (méthode legacy)\n")
        f.write("Spectres: mise à l’échelle 0–1 : y01 = (y - ymin)/(ymax - ymin)\n\n")
        f.write(f"Concentration exclue du fit: {sorted(list(EXCLUDE_CONCENTRATIONS))}\n")
        f.write(f"Bande − : {band_neg[0]:.1f}–{band_neg[1]:.1f} nm\n")
        f.write(f"Bande + : {band_pos[0]:.1f}–{band_pos[1]:.1f} nm\n")
        f.write(f"Calibration: C = {a:.6g} * R + {b:.6g}\n")
        f.write(f"R² = {r2:.6f}\n\n")

        f.write(f"Valeurs fabricant (tolérance ±{int(MFG_REL_TOL*100)}% sur l’étiquette):\n")
        for k, (sg, vm) in MANUFACTURER_LABELS.items():
            ct, dct = manufacturer_percent(sg, vm, rel_tol=MFG_REL_TOL)
            f.write(f"- {k}: {ct:.3f} ± {dct:.3f} %\n")

    print("\n=== Résumé ===")
    print(f"Bande − : {band_neg[0]:.1f}–{band_neg[1]:.1f} nm")
    print(f"Bande + : {band_pos[0]:.1f}–{band_pos[1]:.1f} nm")
    print(f"C = {a:.4g} R + {b:.4g} | R² = {r2:.4f}")
    print(f"Sortie: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()