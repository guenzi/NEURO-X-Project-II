# plotting.py
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Iterable, Tuple, List


# =====================================================================
# Sauvegarde des figures (au lieu de plt.show()) dans results/<date_heure>/
# — identique a Code_mono_stim_v2/plotting.py
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
# Traces temporelles — comme Code_mono_stim_v2, avec DEUX canaux de stimulus (stim1=whisker,
# stim2=auditif) au lieu d'un seul, plus le panneau "Expectation No-Go" general.
# =====================================================================
def plot_traces(time_vect,
                mouse_session,
                reward_array,
                stim1_array=None,
                stim2_array=None,
                title: str = "",
                save_name: Optional[str] = None,
                threshold: float = 1.0):
    def _to_1d(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        return a

    T = len(time_vect)
    expc = _to_1d(mouse_session.expectation)
    nogo = _to_1d(getattr(mouse_session, "expectation_nogo", None))
    plick = _to_1d(mouse_session.p_lick)
    lick = _to_1d(mouse_session.lick)
    rpe = _to_1d(mouse_session.rpe)
    motiv = _to_1d(mouse_session.motivation)
    uncert = _to_1d(getattr(mouse_session, "uncertainty", None))
    reward = _to_1d(reward_array)
    stim1 = _to_1d(stim1_array) if stim1_array is not None else None
    stim2 = _to_1d(stim2_array) if stim2_array is not None else None

    def _align(y):
        if y is None:
            return None
        if len(y) != T:
            n = min(T, len(y))
            return y[:n]
        return y

    expc, nogo, plick, lick, rpe, motiv, uncert, reward, stim1, stim2 = (
        _align(x) for x in (expc, nogo, plick, lick, rpe, motiv, uncert, reward, stim1, stim2)
    )

    rows = [("plot", time_vect, expc, "Expectation")]
    # Panneau No-Go seulement s'il contient un signal (delearning_enable=True quelque part
    # dans la session) — sinon ce serait une ligne plate a 0 inutile.
    if nogo is not None and np.any(nogo):
        rows.append(("plot", time_vect, nogo, "Expectation No-Go"))

    rows.append(("plot", time_vect, plick, "P(Lick)"))
    rows.append(("step", time_vect, lick, "Lick"))
    rows.append(("stem", time_vect, reward, "Reward"))

    has_stim = (stim1 is not None and np.any(stim1)) or (stim2 is not None and np.any(stim2))
    if has_stim:
        rows.append(("stem", time_vect, stim1, "Stim1 (whisker)"))
        rows.append(("stem", time_vect, stim2, "Stim2 (auditif)"))
    else:
        rows.append(("plot", time_vect, rpe, "RPE"))

    rows.append(("plot", time_vect, motiv, "Motivation"))
    if uncert is not None:
        rows.append(("plot", time_vect, uncert, "Uncertainty"))

    fig, axs = plt.subplots(len(rows), 1, figsize=(12, 2.2 * len(rows)), sharex=True)
    if len(rows) == 1:
        axs = [axs]

    fixed_ylim = {
        "Expectation": (0.0, 1.0),
        "Expectation No-Go": (0.0, 1.0),
        "Motivation": (0.0, 1.0),
        "RPE": (-1.05, 1.05),
        "Lick": (-0.05, 1.05),
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
# HR(stim1)/HR(stim2)/FA par blocs a l'interieur d'une session
# =====================================================================
def plot_wdt_block_rates(performance: np.ndarray,
                         wdt_session_or_params,
                         max_trials: int = 400,
                         title: str = "",
                         save_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """performance: colonnes [t, stim_code(0/1/2), lick_detected, latency, reward_detected, outcome]."""
    n = min(len(performance), max_trials)
    if n == 0:
        raise ValueError("performance est vide.")

    block = max(1, int(getattr(wdt_session_or_params, "block_numb", 5)))

    xs, hr1s, hr2s, fas = [], [], [], []
    for start in range(0, n, block):
        end = min(start + block, n)
        blk = performance[start:end, :]

        s1_mask = blk[:, 1] == 1
        s2_mask = blk[:, 1] == 2
        catch_mask = blk[:, 1] == 0

        hr1 = np.sum((blk[:, 5] == 1) & s1_mask) / np.sum(s1_mask) if np.sum(s1_mask) > 0 else np.nan
        hr2 = np.sum((blk[:, 5] == 1) & s2_mask) / np.sum(s2_mask) if np.sum(s2_mask) > 0 else np.nan
        far = np.sum((blk[:, 5] == 3) & catch_mask) / np.sum(catch_mask) if np.sum(catch_mask) > 0 else np.nan

        xs.append(start + (end - start) / 2.0)
        hr1s.append(hr1); hr2s.append(hr2); fas.append(far)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, hr1s, linestyle="None", marker="x", label="Hit rate stim1 (whisker)")
    ax.plot(xs, hr2s, linestyle="None", marker="^", label="Hit rate stim2 (auditif)")
    ax.plot(xs, fas, linestyle="None", marker="o", label="False alarm (catch)")
    ax.set_xlim(0, max_trials); ax.set_ylim(0, 1)
    ax.set_xlabel("Trial #"); ax.set_ylabel("P(Lick)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); _save_fig(fig, category="block_rates", name=save_name or title or "block_rates")

    return np.array(xs), np.array(hr1s), np.array(hr2s), np.array(fas)


# =====================================================================
# HR(stim1)/HR(stim2)/FA — utilitaire + courbes vs sessions
# =====================================================================
def session_rates(perf: Optional[np.ndarray], zero_when_empty: bool = True) -> Tuple[float, float, float]:
    """Calcule (HR_stim1, HR_stim2, FA) sur toute la session."""
    if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
        return (0.0, 0.0, 0.0) if zero_when_empty else (np.nan, np.nan, np.nan)

    s1_mask = perf[:, 1] == 1
    s2_mask = perf[:, 1] == 2
    catch_mask = perf[:, 1] == 0

    def _rate(mask, outcome_code):
        n = int(np.sum(mask))
        if n == 0:
            return 0.0 if zero_when_empty else np.nan
        return float(np.sum((perf[:, 5] == outcome_code) & mask) / n)

    hr1 = _rate(s1_mask, 1)
    hr2 = _rate(s2_mask, 1)
    fa = _rate(catch_mask, 3)
    return hr1, hr2, fa


def plot_session_rates(performance_list: Iterable[Optional[np.ndarray]],
                       session_labels: List[str],
                       title: str = "HR(stim1/stim2) & FA(catch) across sessions",
                       zero_when_empty: bool = True,
                       save_name: Optional[str] = None,
                       switch_session: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X = sessions ; Y = P(Lick) pour HR(stim1), HR(stim2), FA(catch). Une ligne verticale
    marque la session ou la contingence bascule (switch_session), si fournie."""
    perfs = list(performance_list)
    if len(session_labels) != len(perfs):
        raise ValueError("session_labels doit avoir la meme longueur que performance_list.")

    rates = [session_rates(p, zero_when_empty=zero_when_empty) for p in perfs]
    hr1_arr = np.array([r[0] for r in rates], dtype=float)
    hr2_arr = np.array([r[1] for r in rates], dtype=float)
    fa_arr = np.array([r[2] for r in rates], dtype=float)
    xs = np.arange(1, len(perfs) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, hr1_arr, linestyle="-", marker="x", color="tab:blue", label="Hit rate stim1 (whisker)")
    ax.plot(xs, hr2_arr, linestyle="-", marker="^", color="tab:orange", label="Hit rate stim2 (auditif)")
    ax.plot(xs, fa_arr, linestyle="-", marker="o", color="tab:green", label="False alarm (catch)")
    if switch_session is not None and 1 <= switch_session <= len(xs):
        ax.axvline(switch_session - 0.5, linestyle="--", color="gray", alpha=0.6, label="switch de contingence")
    ax.set_ylim(0, 1); ax.set_xticks(xs); ax.set_xticklabels(session_labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("P(Lick)")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)
    return xs, hr1_arr, hr2_arr, fa_arr


def plot_single_session_rates(
    perf: Optional[np.ndarray],
    title: str = "HR(stim1/stim2) vs FA (session)",
    zero_when_empty: bool = True,
    save_name: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Plot compact pour UNE session (HR stim1, HR stim2, FA)."""
    hr1, hr2, fa = session_rates(perf, zero_when_empty=zero_when_empty)

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = np.array([1, 2, 3], dtype=float)
    ax.plot([xs[0]], [hr1], linestyle="None", marker="x", color="tab:blue", label="HR stim1")
    ax.plot([xs[1]], [hr2], linestyle="None", marker="^", color="tab:orange", label="HR stim2")
    ax.plot([xs[2]], [fa], linestyle="None", marker="o", color="tab:green", label="FA")
    ax.set_xticks(xs); ax.set_xticklabels(["stim1", "stim2", "FA"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Lick)"); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(); _save_fig(fig, category="session_summary", name=save_name or title)
    return float(hr1), float(hr2), float(fa)


# =====================================================================
# Evolution des gains de stimulus (Go et No-Go) — LE plot qui montre l'extinction
# =====================================================================
def plot_stim_gains(wdt_bundle, config, title: str = "Evolution des gains de stimuli",
                    save_name: Optional[str] = None) -> None:
    """
    Pour chaque session, trace le gain Go de stim1/stim2 (toujours), et si le
    desapprentissage est actif, le gain No-Go de chacun ET le gain NET (go - nogo) —
    c'est ce gain net qui doit s'effondrer pour le stimulus qui cesse d'etre recompense
    apres le switch de contingence, et grimper pour celui qui prend le relais.
    """
    labels = wdt_bundle.labels
    xs = np.arange(1, len(labels) + 1)
    g1 = np.array([r.stim1_gain for r in wdt_bundle.results])
    g2 = np.array([r.stim2_gain for r in wdt_bundle.results])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if config.DELEARNING:
        ng1 = np.array([r.stim1_nogo_gain for r in wdt_bundle.results])
        ng2 = np.array([r.stim2_nogo_gain for r in wdt_bundle.results])
        net1 = g1 - ng1
        net2 = g2 - ng2
        ax.plot(xs, g1, linestyle=":", marker=".", color="tab:blue", alpha=0.5, label="Gain Go stim1")
        ax.plot(xs, ng1, linestyle=":", marker=".", color="tab:red", alpha=0.5, label="Gain No-Go stim1")
        ax.plot(xs, net1, linestyle="-", marker="x", color="tab:blue", linewidth=2, label="Gain NET stim1 (whisker)")
        ax.plot(xs, g2, linestyle=":", marker=".", color="tab:orange", alpha=0.5, label="Gain Go stim2")
        ax.plot(xs, ng2, linestyle=":", marker=".", color="tab:brown", alpha=0.5, label="Gain No-Go stim2")
        ax.plot(xs, net2, linestyle="-", marker="^", color="tab:orange", linewidth=2, label="Gain NET stim2 (auditif)")
    else:
        ax.plot(xs, g1, linestyle="-", marker="x", color="tab:blue", label="Gain stim1 (whisker)")
        ax.plot(xs, g2, linestyle="-", marker="^", color="tab:orange", label="Gain stim2 (auditif)")

    ax.axvline(config.SWITCH_SESSION - 0.5, linestyle="--", color="gray", alpha=0.6, label="switch de contingence")
    ax.axhline(0.0, linestyle="-", color="black", alpha=0.3, linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("Gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)


# =====================================================================
# RPE par lick (scatter) — identique a Code_mono_stim_v2
# =====================================================================
def plot_rpe_per_lick(rpe_table: np.ndarray, title: str = "RPE per lick",
                      save_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if rpe_table is None or rpe_table.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(title); ax.set_xlabel("Lick #"); ax.set_ylabel("RPE")
        ax.text(0.5, 0.5, "No usable licks", transform=ax.transAxes, ha="center", va="center", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
        return np.array([]), np.array([])

    x = rpe_table[:, 0]
    y = rpe_table[:, 2]
    rewarded = rpe_table[:, 3] > 0.5

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x[~rewarded], y[~rewarded], linestyle="None", marker="x", label="No reward")
    ax.plot(x[rewarded], y[rewarded], linestyle="None", marker="o", label="Reward")
    ax.axhline(0.0, linestyle="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lick #"); ax.set_ylabel("RPE (reward[i+1] − expectation[i])")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
    return x, y


def plot_abs_rpe_per_lick(rpe_table: np.ndarray, title: str = "Absolute RPE per lick",
                          save_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if rpe_table is None or rpe_table.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(title); ax.set_xlabel("Lick #"); ax.set_ylabel("|RPE|")
        ax.text(0.5, 0.5, "No usable licks", transform=ax.transAxes, ha="center", va="center", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
        return np.array([]), np.array([])

    x = rpe_table[:, 0]
    y_abs = np.abs(rpe_table[:, 2])
    rewarded = rpe_table[:, 3] > 0.5

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x[~rewarded], y_abs[~rewarded], linestyle="None", marker="x", label="No reward")
    ax.plot(x[rewarded], y_abs[rewarded], linestyle="None", marker="o", label="Reward")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Lick #"); ax.set_ylabel("|RPE|")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    plt.tight_layout(); _save_fig(fig, category="rpe", name=save_name or title)
    return x, y_abs


# =====================================================================
# Fichier texte recapitulatif des parametres numeriques de la simulation
# =====================================================================
def save_parameters_txt(rows: List[Tuple[str, Optional[object]]], filename: str = "parameters.txt") -> str:
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
