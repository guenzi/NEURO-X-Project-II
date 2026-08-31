# plotting.py
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from typing import Optional, Iterable, Tuple, List


# =====================================================================
# Sauvegarde des figures (au lieu de plt.show()) dans results/<date_heure>/
# =====================================================================
_SAVE_DIR: Optional[str] = None
_FIG_COUNT: int = 0


def init_results_dir(base_dir: Optional[str] = None, suffix: str = "") -> str:
    """Cree (si besoin) results/<date_heure>[_suffix]/ et fixe le dossier de sortie pour ce run."""
    global _SAVE_DIR, _FIG_COUNT
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d_%Hh%M")
    folder_name = f"{timestamp}_{suffix}" if suffix else timestamp
    _SAVE_DIR = os.path.join(base_dir, folder_name)
    os.makedirs(_SAVE_DIR, exist_ok=True)
    _FIG_COUNT = 0
    return _SAVE_DIR


def _slugify(text: str) -> str:
    text = text.strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "—", "–"):
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "plot"


def _save_fig(fig, category: str = "", name: str = "") -> str:
    """Sauvegarde fig dans results/<date_heure>/<category>/, puis la ferme."""
    global _FIG_COUNT
    if _SAVE_DIR is None:
        init_results_dir()
    out_dir = os.path.join(_SAVE_DIR, category) if category else _SAVE_DIR
    os.makedirs(out_dir, exist_ok=True)
    _FIG_COUNT += 1
    filename = f"{_FIG_COUNT:03d}_{_slugify(name)}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
# Traces temporelles (expectation, p_lick, lick, reward, stim/RPE, motivation)
# =====================================================================
def plot_traces(time_vect,
                mouse_session,
                reward_array,
                stim_array=None,
                title: str = "",
                with_slider: bool = False,   # conservé pour compat, ignoré
                noise_trace=None,
                save_name: Optional[str] = None,
                threshold: float = 1.0):
    """
    Affiche les traces temporelles:
      Expectation, P(Lick), Lick, Reward, Stim (ou RPE), Motivation
      + Noise gain (si noise_trace est fourni)
    - time_vect: (T,)
    - reward_array, stim_array, noise_trace: (T,1) ou (T,)
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    def _to_1d(arr):
        if arr is None:
            return None
        a = _np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        return a

    T = len(time_vect)
    expc   = _to_1d(mouse_session.expectation)
    plick  = _to_1d(mouse_session.p_lick)
    lick   = _to_1d(mouse_session.lick)
    rpe    = _to_1d(mouse_session.rpe)
    motiv  = _to_1d(mouse_session.motivation)
    uncert = _to_1d(getattr(mouse_session, "uncertainty", None))
    reward = _to_1d(reward_array)
    stim   = _to_1d(stim_array) if stim_array is not None else None
    noise  = _to_1d(noise_trace) if noise_trace is not None else None

    # sécurité: tronque si nécessaire pour s'aligner à time_vect
    def _align(y):
        if y is None:
            return None
        if len(y) != T:
            n = min(T, len(y))
            return y[:n]
        return y

    expc   = _align(expc)
    plick  = _align(plick)
    lick   = _align(lick)
    rpe    = _align(rpe)
    motiv  = _align(motiv)
    uncert = _align(uncert)
    reward = _align(reward)
    stim   = _align(stim)
    noise  = _align(noise)

    # Construction des panneaux
    rows = [
        ("plot", time_vect, expc,   "Expectation"),
        ("plot", time_vect, plick,  "P(Lick)"),
        ("step", time_vect, lick,   "Lick"),
        ("stem", time_vect, reward, "Reward"),
    ]

    if stim is not None and _np.any(stim):
        rows.append(("plot", time_vect, stim, "Stim"))
    else:
        rows.append(("plot", time_vect, rpe,  "RPE"))

    rows.append(("plot", time_vect, motiv, "Motivation"))

    if uncert is not None:
        rows.append(("plot", time_vect, uncert, "Uncertainty"))

    # Ajoute le panneau Noise gain si dispo
    if noise is not None:
        rows.append(("plot", time_vect, noise, "Noise gain"))

    # Figure
    fig, axs = _plt.subplots(len(rows), 1, figsize=(12, 2.2*len(rows)), sharex=True)

    # Bornes fixes pour les grandeurs dont la plage est definie par le modele
    # (evite qu'un auto-scale sur une toute petite variation ne trompe la lecture)
    fixed_ylim = {
        "Expectation": (0.0, 1.0),
        "Motivation": (0.0, 1.0),
        "RPE": (-1.05, 1.05),
        "Lick": (-0.05, 1.05),
        # Uncertainty n'a pas de bornes fixes: contrairement aux autres grandeurs,
        # sa plage reellement atteinte varie beaucoup d'une session a l'autre (souvent
        # tres inferieure a U_max=1.0) — un axe fixe [0,1] ecraserait la variation utile.
    }

    for ax, (kind, x, y, label) in zip(axs, rows):
        if kind == "plot":
            ax.plot(x, y)
            if label == "P(Lick)":
                ax.axhline(threshold, linestyle="--", alpha=0.5)
        elif kind == "step":
            ax.step(x, y, where="post")
        elif kind == "stem":
            ax.stem(x, y, linefmt='-', markerfmt=' ', basefmt=' ')
        ax.set_ylabel(label)
        if label in fixed_ylim:
            ax.set_ylim(*fixed_ylim[label])

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    _save_fig(fig, category="traces", name=save_name or title)


# =====================================================================
# Value / Cost par trial (brut) + moyenne glissante (expected_V, expected_C)
# =====================================================================
def plot_value_cost_trials(
    trial_times: List[float],
    v_trials: List[float],
    c_trials: List[float],
    expected_v: List[float],
    expected_c: List[float],
    title: str = "Value / Cost per trial",
    save_name: Optional[str] = None,
) -> None:
    """
    Deux courbes (expected_V, expected_C lisses, moyenne glissante) et les points
    bruts de chaque trial (v_trials, c_trials) par-dessus.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(trial_times, expected_v, "-", color="tab:blue", linewidth=1.6, label="Expected V (smoothed)")
    ax.plot(trial_times, expected_c, "-", color="tab:orange", linewidth=1.6, label="Expected C (smoothed)")

    ax.scatter(trial_times, v_trials, marker="o", s=14, color="tab:blue", alpha=0.45, label="V (trial)")
    ax.scatter(trial_times, c_trials, marker="o", s=14, color="tab:orange", alpha=0.45, label="C (trial)")

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value / Cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    fig.tight_layout()
    _save_fig(fig, category="traces", name=save_name or title)


# =====================================================================
# HR & FA moyens par session (liste)
# =====================================================================
def plot_hr_fa_over_sessions(performance_list: List[np.ndarray],
                             labels: List[str],
                             title: str = "HR & FA average per session",
                             save_name: Optional[str] = None):
    """
    performance_list: list of np.ndarray, chacun = sortie de function_performance_wdt pour une session
                      colonnes: [t, stim_amp, lick_detected, latency, reward_detected, outcome]
    labels:           list[str], ex. ["WDT1", "WDT2", ...]
    """
    hr_vals, fa_vals = [], []

    for perf in performance_list:
        if perf is None or perf.size == 0:
            hr_vals.append(np.nan)
            fa_vals.append(np.nan)
            continue

        stim_trials = perf[:, 1] > 0
        catch_trials = perf[:, 1] == 0
        hit_trials = perf[:, 5] == 1
        fa_trials = perf[:, 5] == 3

        hr = np.sum(hit_trials & stim_trials) / np.sum(stim_trials) if np.sum(stim_trials) > 0 else np.nan
        fa = np.sum(fa_trials & catch_trials) / np.sum(catch_trials) if np.sum(catch_trials) > 0 else np.nan

        hr_vals.append(hr)
        fa_vals.append(fa)

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, hr_vals, 'o-', label='Hit Rate (HR)')
    plt.plot(x, fa_vals, 'o-', label='False Alarm (FA)')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel('Rate')
    plt.xlabel('Session')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save_fig(fig, category="learning_curve", name=save_name or title)


# =====================================================================
# HR/FA par blocs à l’intérieur d’une session
# =====================================================================
def plot_wdt_block_rates(performance: np.ndarray,
                         wdt_params,
                         max_trials: int = 400,
                         title: str = "",
                         save_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    performance: np.ndarray renvoyé par function_performance_wdt
                 colonnes = [t, stim_amp, lick_detected, latency, reward_detected, outcome]
                 outcome: 0=Miss, 1=Hit, 2=CR, 3=FA
    wdt_params: WDTSesssionParams (utilise .block_numb)
    max_trials: borne X (ex: 400)
    """
    n = min(len(performance), max_trials)
    if n == 0:
        raise ValueError("performance est vide.")

    block = max(1, int(getattr(wdt_params, "block_numb", 5)))

    xs, hrs, fas = [], [], []
    for start in range(0, n, block):
        end = min(start + block, n)
        blk = performance[start:end, :]

        stim_mask = blk[:, 1] > 0
        catch_mask = ~stim_mask

        stim_cnt = int(np.sum(stim_mask))
        catch_cnt = int(np.sum(catch_mask))

        hits = int(np.sum((blk[:, 5] == 1) & stim_mask))
        fas_ = int(np.sum((blk[:, 5] == 3) & catch_mask))

        hr = hits / stim_cnt if stim_cnt > 0 else np.nan
        far = fas_ / catch_cnt if catch_cnt > 0 else np.nan

        xs.append(start + (end - start) / 2.0)
        hrs.append(hr); fas.append(far)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, hrs, linestyle="None", marker="x", label="Hit rate (block)")
    ax.plot(xs, fas, linestyle="None", marker="o", label="False alarm (block)")
    ax.set_xlim(0, max_trials); ax.set_ylim(0, 1)
    ax.set_xlabel("Trial # (stim & catch)"); ax.set_ylabel("P(Lick)")
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); _save_fig(fig, category="block_rates", name=save_name or title or "block_rates")

    return np.array(xs), np.array(hrs), np.array(fas)


# =====================================================================
# HR, FA, d′ — utilitaire + courbes vs sessions
# =====================================================================
def session_rates(perf: Optional[np.ndarray],
                  zero_when_empty: bool = True) -> Tuple[float, float, float]:
    """
    Calcule (HR, FA, d′) sur toute la session à partir de 'performance'
    """
    if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
        return (0.0, 0.0, np.nan) if zero_when_empty else (np.nan, np.nan, np.nan)

    stim_mask = perf[:, 1] > 0
    catch_mask = ~stim_mask

    stim_n = int(np.sum(stim_mask))
    catch_n = int(np.sum(catch_mask))

    hits = int(np.sum((perf[:, 5] == 1) & stim_mask))
    fas = int(np.sum((perf[:, 5] == 3) & catch_mask))

    hr = (hits / stim_n) if stim_n > 0 else (0.0 if zero_when_empty else np.nan)
    fa = (fas  / catch_n) if catch_n > 0 else (0.0 if zero_when_empty else np.nan)

    if stim_n == 0 or catch_n == 0 or np.isnan(hr) or np.isnan(fa):
        dprime = float(np.nan)
    else:
        hr_corr = min(max(float(hr), 0.01), 0.99)
        fa_corr = min(max(float(fa), 0.01), 0.99)
        nd = NormalDist()
        dprime = float(nd.inv_cdf(hr_corr) - nd.inv_cdf(fa_corr))

    return float(hr), float(fa), dprime


def plot_session_rates(performance_list: Iterable[Optional[np.ndarray]],
                       session_labels: List[str],
                       title: str = "HR, FA & d′ per session (WDT)",
                       zero_when_empty: bool = True,
                       ax: Optional[plt.Axes] = None,
                       save_name: Optional[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X = sessions ; Y (gauche) = P(Lick) pour HR/FA ; Y (droite) = d′.
    Retourne (xs, hr_arr, fa_arr, dp_arr).
    """
    perfs = list(performance_list)
    if len(session_labels) != len(perfs):
        raise ValueError("session_labels doit avoir la même longueur que performance_list.")

    rates: List[Tuple[float, float, float]] = [
        session_rates(p, zero_when_empty=zero_when_empty) for p in perfs
    ]
    hr_arr = np.array([hr for hr, _, _ in rates], dtype=float)
    fa_arr = np.array([fa for _, fa, _ in rates], dtype=float)
    dp_arr = np.array([dp for _, _, dp in rates], dtype=float)
    xs = np.arange(1, len(perfs) + 1)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        created_fig = True

    ax.plot(xs, hr_arr, linestyle="None", marker="x", label="Hit rate")
    ax.plot(xs, fa_arr, linestyle="None", marker="o", label="False alarm")
    ax.set_ylim(0, 1); ax.set_xticks(xs); ax.set_xticklabels(session_labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("P(Lick)")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)

    ax_d = ax.twinx()
    ax_d.plot(xs, dp_arr, linestyle="--", marker="o", color="green", label="d′")
    ax_d.set_ylabel("d′"); ax_d.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_d.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

    if created_fig:
        plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)

    return xs, hr_arr, fa_arr, dp_arr


# =====================================================================
# Plots compacts (une session / multi-souris)
# =====================================================================
def plot_single_session_rates(
    perf: Optional[np.ndarray],
    title: str = "HR vs FA (session)",
    zero_when_empty: bool = True,
    ax: Optional[plt.Axes] = None,
    save_name: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Plot compact pour UNE session (HR & FA) + d′ en légende."""
    hr, fa, dprime = session_rates(perf, zero_when_empty=zero_when_empty)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True

    xs = np.array([1, 2], dtype=float)
    ax.plot([xs[0]], [hr], linestyle="None", marker="x", label="HR")
    ax.plot([xs[1]], [fa], linestyle="None", marker="o", label="FA")

    ax.set_ylim(0, 1)
    ax.set_xticks(xs); ax.set_xticklabels(["HR", "FA"])
    ax.set_ylabel("P(Lick)"); ax.set_title(title)

    dprime_txt = f"d′ = {dprime:.2f}" if not np.isnan(dprime) else "d′ : n/a"
    leg = ax.legend(loc="upper right", frameon=True, title=dprime_txt)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_alpha(0.9)
    ax.grid(True, axis="y", alpha=0.3)

    if created_fig:
        plt.tight_layout(); _save_fig(fig, category="session_summary", name=save_name or title)

    return float(hr), float(fa), float(dprime)


def plot_multi_mouse_single_session(
    performance_list: Iterable[Optional[np.ndarray]],
    mouse_labels: List[str],
    title: str = "HR vs FA per mouse (single session)",
    zero_when_empty: bool = True,
    jitter: float = 0.06,
    annotate_dprime: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Superpose HR (x) et FA (o) pour plusieurs souris sur UNE session.
    X = ["HR","FA"], Y = P(Lick)∈[0,1]. d′ annoté près de HR.
    """
    perfs = list(performance_list)
    if len(perfs) != len(mouse_labels):
        raise ValueError("mouse_labels doit avoir la même longueur que performance_list.")

    n = len(perfs)
    hr_arr = np.empty(n, dtype=float)
    fa_arr = np.empty(n, dtype=float)
    dp_arr = np.empty(n, dtype=float)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        created_fig = True

    x_hr, x_fa = 1.0, 2.0
    offs = np.linspace(-jitter, jitter, n) if n > 1 else np.array([0.0])

    for i, (perf, label) in enumerate(zip(perfs, mouse_labels)):
        hr, fa, dp = session_rates(perf, zero_when_empty=zero_when_empty)
        hr_arr[i], fa_arr[i], dp_arr[i] = hr, fa, dp

        (line_hr,) = ax.plot([x_hr + offs[i]], [hr], linestyle="None", marker="x",
                             label=f"{label} — d′={dp:.2f}")
        color = line_hr.get_color()
        ax.plot([x_fa + offs[i]], [fa], linestyle="None", marker="o",
                color=color, label="_nolegend_")

        if annotate_dprime:
            ax.annotate(f"{dp:.2f}",
                        xy=(x_hr + offs[i], hr),
                        xytext=(x_hr + offs[i] + 0.03, min(1.0, hr + 0.06)),
                        fontsize=9, color=color)

    ax.set_ylim(0, 1)
    ax.set_xticks([x_hr, x_fa]); ax.set_xticklabels(["HR", "FA"])
    ax.set_ylabel("P(Lick)"); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncols=1, frameon=True)

    if created_fig and show:
        plt.tight_layout(); _save_fig(fig, category="population", name=save_name or title)
    return hr_arr, fa_arr, dp_arr


def overlay_group_stats_on_current_axes(hr_arr: np.ndarray,
                                        fa_arr: np.ndarray,
                                        dp_arr: np.ndarray,
                                        label: str = "Mean ± SD",
                                        session_name: Optional[str] = None,
                                        update_title: bool = True,
                                        show_errorbars: bool = True) -> None:
    """
    Superpose sur la figure courante:
      - HR_mean ± SD et FA_mean ± SD (optionnel)
      - d′ moyen annoté
      - Encadré avec μ, σ, σ² pour HR & FA
    À appeler juste après plot_multi_mouse_single_session(...).
    """
    ax = plt.gca()
    x_hr, x_fa = 1.0, 2.0

    hr_mean = float(np.nanmean(hr_arr)) if hr_arr.size else np.nan
    fa_mean = float(np.nanmean(fa_arr)) if fa_arr.size else np.nan
    dp_mean = float(np.nanmean(dp_arr)) if dp_arr.size else np.nan

    hr_sd = float(np.nanstd(hr_arr, ddof=1)) if hr_arr.size > 1 else 0.0
    fa_sd = float(np.nanstd(fa_arr, ddof=1)) if fa_arr.size > 1 else 0.0

    hr_var = float(np.nanvar(hr_arr, ddof=1)) if hr_arr.size > 1 else 0.0
    fa_var = float(np.nanvar(fa_arr, ddof=1)) if fa_arr.size > 1 else 0.0

    if show_errorbars:
        ax.errorbar([x_hr], [hr_mean], yerr=[hr_sd], fmt='s', ms=8, capsize=4, label=f"{label} HR")
        ax.errorbar([x_fa], [fa_mean], yerr=[fa_sd], fmt='s', ms=8, capsize=4, label=f"{label} FA")

    if np.isfinite(dp_mean):
        ax.annotate(f"mean d′ = {dp_mean:.2f}",
                    xy=(x_hr, hr_mean),
                    xytext=(x_hr + 0.08, min(1.0, hr_mean + 0.10)),
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

    stats_lines = [
        f"HR: μ={hr_mean:.2f}, σ={hr_sd:.2f}, σ²={hr_var:.3f}",
        f"FA: μ={fa_mean:.2f}, σ={fa_sd:.2f}, σ²={fa_var:.3f}",
    ]
    if np.isfinite(dp_mean):
        stats_lines.append(f"d′ mean: {dp_mean:.2f}")

    ax.text(0.02, 0.98, "\n".join(stats_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="gray", alpha=0.9))

    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    if session_name and update_title:
        old_title = ax.get_title()
        if old_title and session_name not in old_title:
            ax.set_title(f"{old_title}\nSession: {session_name}")
        elif not old_title:
            ax.set_title(f"Session: {session_name}")


# =====================================================================
# RPE par lick (scatter)
# =====================================================================
def plot_rpe_per_lick(rpe_table: np.ndarray,
                      title: str = "RPE per lick",
                      ax: Optional[plt.Axes] = None,
                      show: bool = True,
                      save_name: Optional[str] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    rpe_table: (N x 6) tel que renvoyé par extract_rpe_per_lick
               colonnes = [lick_idx, t_sec, rpe, reward_flag, stim_amp, bin_index]
    """
    if rpe_table is None or rpe_table.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4)) if ax is None else (None, ax)
        ax.set_title(title); ax.set_xlabel("Lick #"); ax.set_ylabel("RPE")
        ax.text(0.5, 0.5, "No usable licks", transform=ax.transAxes,
                ha="center", va="center", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.3)
        if fig is not None and show:
            plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
        return np.array([]), np.array([])

    x = rpe_table[:, 0]
    y = rpe_table[:, 2]
    rewarded = rpe_table[:, 3] > 0.5

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        created_fig = True

    ax.plot(x[~rewarded], y[~rewarded], linestyle="None", marker="x", label="No reward")
    ax.plot(x[rewarded],  y[rewarded],  linestyle="None", marker="o", label="Reward")

    ax.axhline(0.0, linestyle="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lick #")
    ax.set_ylabel("RPE (reward[i+1] − expectation[i])")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    if created_fig and show:
        plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)

    return x, y

def plot_abs_rpe_per_lick(rpe_table: np.ndarray,
                          title: str = "Absolute RPE per lick",
                          ax: Optional[plt.Axes] = None,
                          show: bool = True,
                          save_name: Optional[str] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    rpe_table: (N x 6) tel que renvoyé par extract_rpe_per_lick
               colonnes = [lick_idx, t_sec, rpe, reward_flag, stim_amp, bin_index]
    Affiche:
      - scatter |RPE| vs index de lick
      - marqueurs différents pour reward vs no-reward
    Retourne (x, y_abs) utiles si besoin.
    """
    if rpe_table is None or rpe_table.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4)) if ax is None else (None, ax)
        ax.set_title(title)
        ax.set_xlabel("Lick #")
        ax.set_ylabel("|RPE|")
        ax.text(0.5, 0.5, "No usable licks", transform=ax.transAxes,
                ha="center", va="center", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.3)
        if fig is not None and show:
            plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
        return np.array([]), np.array([])

    x = rpe_table[:, 0]
    y_abs = np.abs(rpe_table[:, 2])
    rewarded = rpe_table[:, 3] > 0.5

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        created_fig = True

    # scatter: récompensé (•), non-récompensé (x)
    ax.plot(x[~rewarded], y_abs[~rewarded], linestyle="None", marker="x", label="No reward")
    ax.plot(x[rewarded],  y_abs[rewarded],  linestyle="None", marker="o", label="Reward")

    # habillage
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Lick #")
    ax.set_ylabel("|RPE|")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    if created_fig and show:
        plt.tight_layout()
        _save_fig(fig, category="rpe", name=save_name or title)

    return x, y_abs

def plot_noise_gain_sigmoid(m0: float,
                            k: float = 8.0,
                            gmin: float = 0.02,
                            gmax: float = 0.15,
                            m_grid: Optional[np.ndarray] = None,
                            ax: Optional[plt.Axes] = None,
                            title: Optional[str] = None,
                            show: bool = True,
                            save_name: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trace la loi statique croissante:
        g(m) = gmin + (gmax - gmin) * sigmoid(k * (m - m0)),
    avec m ∈ [0,1] = mean(|RPE|) sur la fenêtre.

    Params clés:
      - m0 : point d'inflexion (ex: 0.60)
      - k  : pente (↑ = transition plus raide)
      - gmin, gmax : bornes du noise gain

    Retourne (m, g) pour réutilisation éventuelle.
    """
    if m_grid is None:
        m_grid = np.linspace(0.0, 1.0, 501)
    m = np.clip(np.asarray(m_grid, dtype=float), 0.0, 1.0)
    g = gmin + (gmax - gmin) * (1.0 / (1.0 + np.exp(-k * (m - m0))))

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        created = True

    ax.plot(m, g, label=f"k={k:g}, m0={m0:g}, [{gmin:g}–{gmax:g}]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min(gmin, gmax), max(gmin, gmax))
    ax.set_xlabel("m = mean(|RPE|) sur les n derniers licks")
    ax.set_ylabel("noise gain g")
    ax.set_title(title or "Static mapping (sigmoid): g(m)")
    # repères
    ax.axvline(m0, linestyle="--", alpha=0.5)
    ax.axhline(gmin + 0.5*(gmax - gmin), linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    if created and show:
        plt.tight_layout(); _save_fig(fig, category="noise", name=save_name or title or "noise_gain_mapping")
    return m, g





def plot_population_hr_fa(
    performance_list: Iterable[Optional[np.ndarray]],
    mouse_labels: List[str],
    session_name: str = "WDT1",
    noise_gain: float = 0.05,
    learning_stim: float = 0.02,
    jitter: float = 0.04,
    annotate_dprime: bool = True,
    zero_when_empty: bool = True,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
):
    """
    Style 'cohorte' façon capture :
      - X = ["HR","FA"], Y in [0,1]
      - HR = croix, FA = points pleins
      - une couleur par souris (HR & FA assortis)
      - d′ annoté près des points HR
      - box stats en haut-gauche + légende à droite
    """
    perfs = list(performance_list)
    if len(perfs) != len(mouse_labels):
        raise ValueError("mouse_labels doit avoir la même longueur que performance_list.")

    n = len(perfs)
    hr_arr = np.empty(n, dtype=float)
    fa_arr = np.empty(n, dtype=float)
    dp_arr = np.empty(n, dtype=float)
    for i, p in enumerate(perfs):
        hr, fa, dp = session_rates(p, zero_when_empty=zero_when_empty)
        hr_arr[i], fa_arr[i], dp_arr[i] = hr, fa, dp

    # Figure
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x_hr, x_fa = 1.0, 2.0
    offs = np.linspace(-jitter, jitter, n) if n > 1 else np.array([0.0])

    legend_handles = []
    for i, label in enumerate(mouse_labels):
        line = ax.plot([], [], marker="x")[0]        # pour obtenir une couleur cohérente
        color = line.get_color()
        line.remove()

        # HR (croix) et FA (point plein) avec même couleur
        ax.plot([x_hr + offs[i]], [hr_arr[i]], linestyle="None", marker="x", markersize=9, color=color)
        ax.plot([x_fa + offs[i]], [fa_arr[i]], linestyle="None", marker="o", markersize=7, color=color)

        if annotate_dprime and np.isfinite(dp_arr[i]):
            ax.annotate(f"{dp_arr[i]:.2f}",
                        xy=(x_hr + offs[i], hr_arr[i]),
                        xytext=(x_hr + offs[i] + 0.03, min(1.0, hr_arr[i] + 0.06)),
                        fontsize=10, color=color)

        legend_handles.append(plt.Line2D([0], [0], linestyle="None",
                                         marker="x", markersize=9, color=color,
                                         label=f"{label} —  d′={dp_arr[i]:.2f}"))

    # Axes, ticks, style
    ax.set_ylim(0, 1)
    ax.set_xlim(0.6, 2.4)
    ax.set_xticks([x_hr, x_fa])
    ax.set_xticklabels(["HR", "FA"], fontsize=12, fontweight="bold")
    ax.set_ylabel("P(Lick)")
    ttl = title or f"{session_name} — HR vs FA (cohort of {n} mice, g={noise_gain:.3f}, lr={learning_stim:.3f})"
    ax.set_title(ttl, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    # Stats groupe (box en haut-gauche)
    hr_mean = float(np.nanmean(hr_arr)); fa_mean = float(np.nanmean(fa_arr))
    dp_mean = float(np.nanmean(dp_arr)) if np.isfinite(dp_arr).any() else np.nan
    hr_sd   = float(np.nanstd(hr_arr, ddof=1)) if n > 1 else 0.0
    fa_sd   = float(np.nanstd(fa_arr, ddof=1)) if n > 1 else 0.0
    hr_var  = float(np.nanvar(hr_arr, ddof=1)) if n > 1 else 0.0
    fa_var  = float(np.nanvar(fa_arr, ddof=1)) if n > 1 else 0.0

    stats_lines = [
        f"HR: μ={hr_mean:.2f}, σ={hr_sd:.2f}, σ²={hr_var:.3f}",
        f"FA: μ={fa_mean:.2f}, σ={fa_sd:.2f}, σ²={fa_var:.3f}",
    ]
    if np.isfinite(dp_mean):
        stats_lines.append(f"d′ mean: {dp_mean:.2f}")

    ax.text(0.02, 0.98, "\n".join(stats_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    # Légende à droite
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, title=None)

    plt.tight_layout()
    _save_fig(fig, category="population", name=save_name or f"population_hr_fa_{session_name}")

    return hr_arr, fa_arr, dp_arr


# =====================================================================
# Fichier texte recapitulatif des parametres numeriques de la simulation
# =====================================================================
def save_parameters_txt(
    rows: List[Tuple[str, Optional[object]]],
    filename: str = "parameters.txt",
) -> str:
    """
    rows: liste de tuples (name, value). Une ligne de section (titre de groupe)
    s'obtient en passant value=None. Ecrit results/<date_heure>/<filename>.
    """
    if _SAVE_DIR is None:
        init_results_dir()
    path = os.path.join(_SAVE_DIR, filename)

    name_width = max((len(str(name)) for name, value in rows if value is not None), default=20)

    with open(path, "w") as f:
        for name, value in rows:
            if value is None:
                f.write(f"\n{name}\n")
                f.write("-" * len(name) + "\n")
            else:
                f.write(f"{str(name):<{name_width}} : {value}\n")

    return path
