# plotting.py
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from typing import Optional, Iterable, Tuple, List


# Sauvegarde des figures (au lieu de plt.show()) dans results/<date_heure>/
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
    # Titre de plot -> nom de fichier propre, utilise par _save_fig si save_name est vide.
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


# Traces temporelles (expectation, p_lick, lick, reward, stim/RPE, motivation)
def plot_traces(time_vect,
                mouse_session,
                reward_array,
                stim_array=None,
                title: str = "",
                with_slider: bool = False,   # conservé pour compat, ignoré
                noise_trace=None,
                save_name: Optional[str] = None,
                threshold: float = 1.0,
                dual_stim: bool = False,
                stim2_array=None):
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
    if dual_stim:
        expc = None
        e_right = _to_1d(mouse_session.expectation_right)
        e_left = _to_1d(mouse_session.expectation_left)
        nogo = None
        nogo_right = _to_1d(getattr(mouse_session, "expectation_nogo_right", None))
        nogo_left = _to_1d(getattr(mouse_session, "expectation_nogo_left", None))
    else:
        expc = _to_1d(mouse_session.expectation)
        e_right = e_left = None
        nogo = _to_1d(getattr(mouse_session, "expectation_nogo", None))
        nogo_right = nogo_left = None
    plick  = _to_1d(mouse_session.p_lick)
    lick   = _to_1d(mouse_session.lick)
    rpe    = _to_1d(mouse_session.rpe)
    motiv  = _to_1d(mouse_session.motivation)
    uncert = _to_1d(getattr(mouse_session, "uncertainty", None))
    reward = _to_1d(reward_array)
    stim   = _to_1d(stim_array) if stim_array is not None else None
    # Dual whisker/auditif uniquement (dual_stim=False, une seule Expectation partagee) :
    # stim_array=stim1, stim2_array=stim2, affiches sur deux lignes separees au lieu d'une
    # seule "Stim". Ignore si dual_stim=True (le dual gauche/droite n'a pas ce concept).
    stim2  = _to_1d(stim2_array) if (stim2_array is not None and not dual_stim) else None
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
    e_right = _align(e_right)
    e_left = _align(e_left)
    nogo   = _align(nogo)
    nogo_right = _align(nogo_right)
    nogo_left  = _align(nogo_left)
    plick  = _align(plick)
    lick   = _align(lick)
    rpe    = _align(rpe)
    motiv  = _align(motiv)
    uncert = _align(uncert)
    reward = _align(reward)
    stim   = _align(stim)
    stim2  = _align(stim2)
    noise  = _align(noise)

    # Construction des panneaux
    if dual_stim:
        rows = [
            ("plot_side", time_vect, e_right, "Expectation right"),
            ("plot_side", time_vect, e_left, "Expectation left"),
        ]
        # Architecture Go/No-Go toujours active : les panneaux No-Go sont toujours affiches.
        if nogo_right is not None:
            rows.append(("plot_side", time_vect, nogo_right, "No-Go Expectation right"))
        if nogo_left is not None:
            rows.append(("plot_side", time_vect, nogo_left, "No-Go Expectation left"))
    else:
        rows = [("plot", time_vect, expc, "Expectation")]
        if nogo is not None:
            rows.append(("plot", time_vect, nogo, "Expectation No-Go"))

    if dual_stim:
        p_right = _np.clip(plick, 0, None) if plick is not None else None
        p_left = _np.clip(-plick, 0, None) if plick is not None else None
        rows.append(("plot_thr", time_vect, p_right, "P(Lick) right"))
        rows.append(("plot_thr", time_vect, p_left, "P(Lick) left"))
    else:
        rows.append(("plot", time_vect, plick, "P(Lick)"))

    rows.append(("step", time_vect, lick, "Lick"))
    rows.append(("stem", time_vect, reward, "Reward"))

    stim_has_signal = stim is not None and _np.any(stim)
    stim2_has_signal = stim2 is not None and _np.any(stim2)
    if stim2 is not None and (stim_has_signal or stim2_has_signal):
        # Dual whisker/auditif : deux lignes separees plutot qu'un seul canal "Stim".
        rows.append(("stem", time_vect, stim, "Stim1 (whisker)"))
        rows.append(("stem", time_vect, stim2, "Stim2 (auditory)"))
    elif stim_has_signal:
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
        "Expectation right": (0.0, 1.0),
        "Expectation left": (0.0, 1.0),
        "Expectation No-Go": (0.0, 1.0),
        "No-Go Expectation right": (0.0, 1.0),
        "No-Go Expectation left": (0.0, 1.0),
        "Motivation": (0.0, 1.0),
        "RPE": (-1.05, 1.05),
        "Lick": (-0.05, 1.05),
        "P(Lick)": (0.0, 1.0),
        "P(Lick) right": (0.0, 1.0),
        "P(Lick) left": (0.0, 1.0),
        "Uncertainty": (0.0, 1.0),
    }

    for ax, (kind, x, y, label) in zip(axs, rows):
        if kind == "plot":
            ax.plot(x, y)
            if label == "P(Lick)":
                ax.axhline(threshold, linestyle="--", alpha=0.5)
        elif kind == "plot_side":
            color = "tab:blue" if "right" in label else "tab:orange"
            ax.plot(x, y, color=color, linewidth=0.8)
        elif kind == "plot_thr":
            color = "tab:blue" if "right" in label else "tab:orange"
            ax.plot(x, y, color=color, linewidth=0.8)
            ax.axhline(threshold, linestyle="--", alpha=0.6, color="gray")
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


# Diagnostic du desapprentissage (Go/No-Go), construit a partir des sessions WDT concatenees
# du pipeline normal.
def plot_delearning_diagnostics(
    session_info,
    mouse_sessions: List,
    labels: List[str],
    delearning_from_session: int,
    perfs: Optional[List] = None,
    dual_stim: bool = False,
    save_prefix: str = "delearning",
    trial_smooth_win: int = 20,
    delearning_until_session: Optional[int] = None,
) -> None:
    # 1. Hit Rate par essai (derniere session recompensee / 1ere extinction / 1ere reacquisition)
    # 2. E_go vs E_nogo dans le temps
    # 3. Hit Rate moyen par session, en barres
    def _to_1d(arr):
        a = np.asarray(arr)
        return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a

    T = session_info.number_bin
    dt = session_info.resolution
    n_sessions = len(mouse_sessions)
    t_full = np.arange(n_sessions * T) * dt
    boundary_t = (delearning_from_session - 1) * T * dt
    reacq_t = (delearning_until_session - 1) * T * dt if delearning_until_session is not None else None

    win = max(1, int(round(30.0 / dt)))  # fenetre de moyenne glissante : 30s

    def smooth(x):
        return np.convolve(x, np.ones(win) / win, mode="same")

    # --- Plot 1 : Hit Rate par essai pour la derniere session recompensee, la 1ere session
    # de desapprentissage, et (si reapprentissage programme) la 1ere session de reapprentissage
    # (2 ou 3 subplots cote a cote), pas le taux de leche brut par bin (qui inclut le lechage
    # spontane entre les essais) ni les essais catch (qui restent proches de 0 et diluent le
    # signal), ni les sessions WDT concatenees (pas utile ici, on veut juste les transitions).
    if perfs is not None:
        panels = [(delearning_from_session - 2, "last rewarded session"),
                  (delearning_from_session - 1, "1st delearning session")]
        if delearning_until_session is not None:
            panels.append((delearning_until_session - 1, "1st relearning session"))

        def _hit_rate_trial_series(perf, side=None):
            # side=None (mono) : tous les essais stimulus. side=+1/-1 (dual) : uniquement
            # les essais de ce cote, hit = outcome 1 (bon cote), pour comparer droite/gauche.
            mask = (perf[:, 1] == side) if side is not None else (perf[:, 1] != 0)
            t = perf[mask, 0]
            resp = (perf[mask, 5] == 1).astype(float) if side is not None else perf[mask, 2].astype(float)
            w = max(1, min(trial_smooth_win, resp.size))
            smoothed = np.convolve(resp, np.ones(w) / w, mode="same") if resp.size else resp
            return t, smoothed, w

        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), sharey=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (s_idx, tag) in zip(axes, panels):
            perf = perfs[s_idx] if 0 <= s_idx < len(perfs) else None
            if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
                continue
            if dual_stim:
                t_r, resp_r, w = _hit_rate_trial_series(perf, side=1)
                t_l, resp_l, _ = _hit_rate_trial_series(perf, side=-1)
                ax.plot(t_r, resp_r, color="tab:blue", label="Hit Rate right")
                ax.plot(t_l, resp_l, color="tab:orange", label="Hit Rate left")
                ax.legend(loc="best", fontsize=8)
            else:
                t, resp_smooth, w = _hit_rate_trial_series(perf)
                ax.plot(t, resp_smooth, color="tab:blue")
            ax.set_xlabel("Time in session (s)")
            ax.set_title(f"{labels[s_idx]} — {tag}")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(f"Hit Rate (rolling average, {trial_smooth_win} trials)")
        axes[0].set_ylim(-0.02, 1.02)
        fig.suptitle("Delearning — Hit Rate before / after reward cutoff")
        fig.tight_layout()
        _save_fig(fig, category="delearning", name=f"{save_prefix}_1_taux_de_leche")

    # --- Plot 2 : E_go vs E_nogo (net) dans le temps ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if dual_stim:
        er = smooth(np.concatenate([_to_1d(ms.expectation_right) for ms in mouse_sessions]))
        el = smooth(np.concatenate([_to_1d(ms.expectation_left) for ms in mouse_sessions]))
        nr = smooth(np.concatenate([_to_1d(ms.expectation_nogo_right) for ms in mouse_sessions]))
        nl = smooth(np.concatenate([_to_1d(ms.expectation_nogo_left) for ms in mouse_sessions]))
        ax.plot(t_full, er - nr, color="tab:blue", label="Net right (E_go − E_nogo)")
        ax.plot(t_full, el - nl, color="tab:orange", label="Net left (E_go − E_nogo)")
    else:
        e_s = smooth(np.concatenate([_to_1d(ms.expectation) for ms in mouse_sessions]))
        n_s = smooth(np.concatenate([_to_1d(ms.expectation_nogo) for ms in mouse_sessions]))
        ax.plot(t_full, e_s, color="tab:blue", label="Expectation (E_go)")
        ax.plot(t_full, n_s, color="tab:red", label="No-Go Expectation")
        ax.plot(t_full, e_s - n_s, color="black", linewidth=1.5, linestyle="--", label="Net (E_go − E_nogo)")
    ax.axvline(boundary_t, linestyle="--", color="gray", alpha=0.7)
    if reacq_t is not None:
        ax.axvline(reacq_t, linestyle="--", color="tab:green", alpha=0.7)
    ax.set_xlabel("Time (s) — concatenated WDT sessions")
    ax.set_ylabel("Value")
    ax.set_title("Delearning — Expectation vs No-Go Expectation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, category="delearning", name=f"{save_prefix}_2_expectation_vs_nogo")

    # --- Plot 3 : Hit Rate moyen par session (barres) ---
    if dual_stim and perfs is not None:
        hr_right = [session_rates_dual(p)[1] for p in perfs]
        hr_left = [session_rates_dual(p)[2] for p in perfs]
        x = np.arange(n_sessions)
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars_r = ax.bar(x - width / 2, hr_right, width, color="tab:blue", label="Hit Rate right")
        bars_l = ax.bar(x + width / 2, hr_left, width, color="tab:orange", label="Hit Rate left")
        for bar, r in zip(list(bars_r) + list(bars_l), hr_right + hr_left):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
        ax.axvline(delearning_from_session - 1.5, linestyle="--", color="gray", alpha=0.7)
        if delearning_until_session is not None:
            ax.axvline(delearning_until_session - 1.5, linestyle="--", color="tab:green", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Average Hit Rate per session")
        ax.set_title("Delearning — Hit Rate right vs left per session")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, category="delearning", name=f"{save_prefix}_3_barres_par_session")
    else:
        if perfs is not None:
            rates = [session_rates(p)[0] for p in perfs]
        else:
            rates = [float(np.mean(_to_1d(ms.lick))) for ms in mouse_sessions]

        def _phase_color(idx):
            s = idx + 1
            if s < delearning_from_session:
                return "tab:green"
            if delearning_until_session is not None and s >= delearning_until_session:
                return "tab:green"
            return "tab:red"

        colors = [_phase_color(idx) for idx in range(n_sessions)]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(labels, rates, color=colors)
        ymax = max(rates) if rates and max(rates) > 0 else 1.0
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, r + ymax * 0.02, f"{r:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("Average Hit Rate per session")
        ax.set_title("Delearning — Average Hit Rate per session")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, category="delearning", name=f"{save_prefix}_3_barres_par_session")


def plot_population_learning_curves(
    records: List[tuple],
    dual_stim: bool = False,
    save_name: str = "population_learning_curves",
    title_suffix: str = "",
    category: str = "population",
) -> None:
    """
    Une courbe de Hit Rate par souris (session_rates / session_rates_dual selon dual_stim),
    une couleur distincte par souris, legende a droite du plot montrant les valeurs des
    parametres qui varient (learning_stim, noise) pour que chaque courbe soit tracable.
    records : liste de (label, learning_stim_i, noise_scale_i, wdt_bundle).
    """
    n = len(records)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(11, 6))
    sessions = None
    for idx, (label, ls_i, ns_i, wdt_bundle) in enumerate(records):
        if dual_stim:
            hrs = [session_rates_dual(r.perf)[0] for r in wdt_bundle.results]
        else:
            hrs = [session_rates(r.perf)[0] for r in wdt_bundle.results]
        sessions = np.arange(1, len(hrs) + 1)
        color = cmap(idx % cmap.N)
        ax.plot(sessions, hrs, color=color, marker="o", markersize=4, linewidth=1.8,
                label=f"{label} — learning_stim={ls_i:.4f}, noise={ns_i:.3f}")

    if sessions is not None:
        ax.set_xticks(sessions)
        ax.set_xticklabels([f"WDT{s}" for s in sessions])
    ax.set_xlabel("Session")
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Population ({n} mice, {'dual' if dual_stim else 'mono'}) — learning curves{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
        title="Variable parameters", title_fontsize=9, frameon=True, borderaxespad=0.0,
    )
    fig.tight_layout()
    _save_fig(fig, category=category, name=save_name)


def plot_population_session_progression(
    records: List[tuple],
    dual_stim: bool = False,
    session_index: int = -1,
    smooth_window_s: float = 600.0,
    save_name: str = "population_session_progression",
    title_suffix: str = "",
    category: str = "population",
) -> None:
    """
    Evolution du Hit Rate AU COURS d'une session WDT (par defaut la derniere, WDT10), par
    souris, lissee sur une fenetre glissante centree de smooth_window_s secondes (~10 min
    par defaut) — pour visualiser l'effet de satiete intra-session (Motivation qui decroit
    au fil de la session, cf. V*M-C) souris par souris. Une couleur par souris, meme legende
    que plot_population_learning_curves (parametres qui varient). En dual, 2 sous-graphes
    cote a cote (droite / gauche), une courbe par souris sur chaque.
    """
    n = len(records)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    half_w = smooth_window_s / 2.0

    def _hr_smooth_series(perf, side=None):
        mask = (perf[:, 1] == side) if side is not None else (perf[:, 1] != 0)
        t = perf[mask, 0]
        hit = (perf[mask, 5] == 1).astype(float)
        order = np.argsort(t)
        t, hit = t[order], hit[order]
        if t.size == 0:
            return t, hit
        hr_smooth = np.array([hit[(t >= ti - half_w) & (t <= ti + half_w)].mean() for ti in t])
        return t, hr_smooth

    session_label = None
    sides = (1, -1) if dual_stim else (None,)
    fig, axes = plt.subplots(1, len(sides), figsize=(9 * len(sides), 6), sharey=True)
    axes = [axes] if len(sides) == 1 else list(axes)

    for ax, side in zip(axes, sides):
        for idx, (label, ls_i, ns_i, wdt_bundle) in enumerate(records):
            if not wdt_bundle.results:
                continue
            result = wdt_bundle.results[session_index]
            session_label = wdt_bundle.labels[session_index]
            perf = result.perf
            if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
                continue

            t, hr_smooth = _hr_smooth_series(perf, side=side)
            if t.size == 0:
                continue

            color = cmap(idx % cmap.N)
            ax.plot(t / 60.0, hr_smooth, color=color, linewidth=1.8,
                    label=f"{label} — learning_stim={ls_i:.4f}, noise={ns_i:.3f}")

        ax.set_xlabel("Time in session (min)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if side is not None:
            ax.set_title("Right" if side == 1 else "Left")

    axes[0].set_ylabel(f"Hit Rate (smoothing ±{smooth_window_s / 60:.0f} min)")
    fig.suptitle(
        f"Population ({n} mice, {'dual' if dual_stim else 'mono'}) — "
        f"Hit Rate progression during {session_label or 'the session'}{title_suffix}"
    )
    axes[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
        title="Variable parameters", title_fontsize=9, frameon=True, borderaxespad=0.0,
    )
    fig.tight_layout()
    _save_fig(fig, category=category, name=save_name)


# Value / Cost par trial (brut) + moyenne glissante (expected_V, expected_C)
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


# ancien modele, non utilise
# # HR & FA moyens par session (liste)
# def plot_hr_fa_over_sessions(performance_list: List[np.ndarray],
#                              labels: List[str],
#                              title: str = "HR & FA average per session",
#                              save_name: Optional[str] = None):
#     """
#     performance_list: list of np.ndarray, chacun = sortie de function_performance_wdt pour une session
#                       colonnes: [t, stim_amp, lick_detected, latency, reward_detected, outcome]
#     labels:           list[str], ex. ["WDT1", "WDT2", ...]
#     """
#     hr_vals, fa_vals = [], []
#
#     for perf in performance_list:
#         if perf is None or perf.size == 0:
#             hr_vals.append(np.nan)
#             fa_vals.append(np.nan)
#             continue
#
#         stim_trials = perf[:, 1] > 0
#         catch_trials = perf[:, 1] == 0
#         hit_trials = perf[:, 5] == 1
#         fa_trials = perf[:, 5] == 3
#
#         hr = np.sum(hit_trials & stim_trials) / np.sum(stim_trials) if np.sum(stim_trials) > 0 else np.nan
#         fa = np.sum(fa_trials & catch_trials) / np.sum(catch_trials) if np.sum(catch_trials) > 0 else np.nan
#
#         hr_vals.append(hr)
#         fa_vals.append(fa)
#
#     x = np.arange(len(labels))
#     fig = plt.figure(figsize=(8, 5))
#     plt.plot(x, hr_vals, 'o-', label='Hit Rate (HR)')
#     plt.plot(x, fa_vals, 'o-', label='False Alarm (FA)')
#     plt.xticks(x, labels)
#     plt.ylim(0, 1)
#     plt.ylabel('Rate')
#     plt.xlabel('Session')
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     _save_fig(fig, category="learning_curve", name=save_name or title)


# HR/FA par blocs à l’intérieur d’une session
def plot_wdt_block_rates(performance: np.ndarray,
                         wdt_params,
                         max_trials: int = 400,
                         title: str = "",
                         save_name: Optional[str] = None,
                         dual_stim: bool = False):
    """
    performance: np.ndarray renvoyé par function_performance_wdt (ou function_performance_wdt_dual)
                 colonnes = [t, stim_amp, lick_detected, latency, reward_detected, outcome, ...]
                 outcome: 0=Miss, 1=Hit, 2=CR, 3=FA (+ 4=Mismatch si dual_stim)
    wdt_params: WDTSesssionParams (utilise .block_numb)
    max_trials: borne X (ex: 400)
    """
    n = min(len(performance), max_trials)
    if n == 0:
        raise ValueError("performance est vide.")

    block = max(1, int(getattr(wdt_params, "block_numb", 5)))

    xs, hrs, fas = [], [], []
    mismatches = [] if dual_stim else None
    for start in range(0, n, block):
        end = min(start + block, n)
        blk = performance[start:end, :]

        stim_mask = (blk[:, 1] != 0) if dual_stim else (blk[:, 1] > 0)
        catch_mask = ~stim_mask

        stim_cnt = int(np.sum(stim_mask))
        catch_cnt = int(np.sum(catch_mask))

        hits = int(np.sum((blk[:, 5] == 1) & stim_mask))
        fas_ = int(np.sum((blk[:, 5] == 3) & catch_mask))

        hr = hits / stim_cnt if stim_cnt > 0 else np.nan
        far = fas_ / catch_cnt if catch_cnt > 0 else np.nan

        xs.append(start + (end - start) / 2.0)
        hrs.append(hr); fas.append(far)

        if dual_stim:
            mm = int(np.sum((blk[:, 5] == 4) & stim_mask))
            mismatches.append(mm / stim_cnt if stim_cnt > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, hrs, linestyle="None", marker="x", label="Hit rate (block)")
    if dual_stim:
        ax.plot(xs, mismatches, linestyle="None", marker="^", label="Mismatch (block)")
    ax.plot(xs, fas, linestyle="None", marker="o", label="False alarm (block)")
    ax.set_xlim(0, max_trials); ax.set_ylim(0, 1)
    ax.set_xlabel("Trial # (stim & catch)"); ax.set_ylabel("P(Lick)")
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); _save_fig(fig, category="block_rates", name=save_name or title or "block_rates")

    if dual_stim:
        return np.array(xs), np.array(hrs), np.array(fas), np.array(mismatches)
    return np.array(xs), np.array(hrs), np.array(fas)


# HR, FA, d′ — utilitaire + courbes vs sessions
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


def session_rates_dual(perf: Optional[np.ndarray],
                       zero_when_empty: bool = True) -> Tuple[float, float, float, float, float, float]:
    """
    Version dual stim de session_rates : Hit Rate separe par cote (droite/gauche), en
    plus du Mismatch (elle a leche, mauvais cote).
    perf : colonnes [t, correct_side, lick_detected, latency, reward_detected, outcome, lick_side_detected]
    outcome : 0=Miss, 1=Hit, 2=CR, 3=FA, 4=Mismatch ; correct_side : +1=droite, -1=gauche, 0=catch
    Retourne (hr, hr_droite, hr_gauche, mismatch, fa, d′). hr = les deux cotes combines
    (garde le meme sens qu'avant, utile pour le calcul du d′).
    """
    empty = (0.0, 0.0, 0.0, 0.0, 0.0, np.nan) if zero_when_empty else (np.nan,) * 5 + (np.nan,)
    if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
        return empty

    stim_mask = perf[:, 1] != 0
    catch_mask = ~stim_mask
    right_mask = perf[:, 1] == 1
    left_mask = perf[:, 1] == -1

    stim_n = int(np.sum(stim_mask))
    catch_n = int(np.sum(catch_mask))
    right_n = int(np.sum(right_mask))
    left_n = int(np.sum(left_mask))

    hits = int(np.sum((perf[:, 5] == 1) & stim_mask))
    hits_right = int(np.sum((perf[:, 5] == 1) & right_mask))
    hits_left = int(np.sum((perf[:, 5] == 1) & left_mask))
    mismatches = int(np.sum((perf[:, 5] == 4) & stim_mask))
    fas = int(np.sum((perf[:, 5] == 3) & catch_mask))

    hr = (hits / stim_n) if stim_n > 0 else (0.0 if zero_when_empty else np.nan)
    hr_right = (hits_right / right_n) if right_n > 0 else (0.0 if zero_when_empty else np.nan)
    hr_left = (hits_left / left_n) if left_n > 0 else (0.0 if zero_when_empty else np.nan)
    mismatch = (mismatches / stim_n) if stim_n > 0 else (0.0 if zero_when_empty else np.nan)
    fa = (fas / catch_n) if catch_n > 0 else (0.0 if zero_when_empty else np.nan)

    if stim_n == 0 or catch_n == 0 or np.isnan(hr) or np.isnan(fa):
        dprime = float(np.nan)
    else:
        hr_corr = min(max(float(hr), 0.01), 0.99)
        fa_corr = min(max(float(fa), 0.01), 0.99)
        nd = NormalDist()
        dprime = float(nd.inv_cdf(hr_corr) - nd.inv_cdf(fa_corr))

    return float(hr), float(hr_right), float(hr_left), float(mismatch), float(fa), dprime


def success_rate_dual(perf: Optional[np.ndarray], zero_when_empty: bool = True) -> float:
    """
    Pourcentage d'essais reussis (dual) : lick droite apres stim droite (Hit droite), lick
    gauche apres stim gauche (Hit gauche), OU pas de lick sur un catch (Correct Rejection) —
    sur l'ensemble des essais (stim + catch). outcome : 1=Hit, 2=CR (voir session_rates_dual).
    """
    if perf is None or not isinstance(perf, np.ndarray) or perf.size == 0:
        return 0.0 if zero_when_empty else float(np.nan)
    total_n = perf.shape[0]
    if total_n == 0:
        return 0.0 if zero_when_empty else float(np.nan)
    success_n = int(np.sum((perf[:, 5] == 1) | (perf[:, 5] == 2)))
    return float(success_n / total_n)


def plot_session_rates(performance_list: Iterable[Optional[np.ndarray]],
                       session_labels: List[str],
                       title: str = "HR, FA & d′ per session (WDT)",
                       zero_when_empty: bool = True,
                       ax: Optional[plt.Axes] = None,
                       save_name: Optional[str] = None,
                       dual_stim: bool = False,
                      ):
    """
    X = sessions ; Y (gauche) = P(Lick) pour HR/FA(/Mismatch) ; Y (droite) = d′ (mono) ou
    % d'essais reussis (dual, Hit droite + Hit gauche + Correct Rejection sur catch).
    En dual_stim, le Hit Rate est separe par cote (droite/gauche) pour reperer un
    eventuel biais ou une difference de vitesse d'apprentissage entre les deux.
    Retourne (xs, hr_arr, fa_arr, dp_arr), ou (xs, hr_arr, fa_arr, dp_arr, mismatch_arr,
    hr_right_arr, hr_left_arr) si dual_stim.
    """
    perfs = list(performance_list)
    if len(session_labels) != len(perfs):
        raise ValueError("session_labels doit avoir la même longueur que performance_list.")

    if dual_stim:
        rates = [session_rates_dual(p, zero_when_empty=zero_when_empty) for p in perfs]
        hr_arr = np.array([r[0] for r in rates], dtype=float)
        hr_right_arr = np.array([r[1] for r in rates], dtype=float)
        hr_left_arr = np.array([r[2] for r in rates], dtype=float)
        mismatch_arr = np.array([r[3] for r in rates], dtype=float)
        fa_arr = np.array([r[4] for r in rates], dtype=float)
        # % d'essais reussis (Hit droite + Hit gauche + Correct Rejection sur catch) a la
        # place du d' — plus lisible que le d' pour le dual (deja 2 Hit Rate + Mismatch + FA
        # separes par cote sur ce meme plot).
        dp_arr = np.array([success_rate_dual(p, zero_when_empty=zero_when_empty) for p in perfs], dtype=float)
    else:
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

    if dual_stim:
        ax.plot(xs, hr_right_arr, linestyle="None", marker="x", color="tab:blue", label="Hit rate right")
        ax.plot(xs, hr_left_arr, linestyle="None", marker="x", color="tab:orange", label="Hit rate left")
        ax.plot(xs, mismatch_arr, linestyle="None", marker="^", color="tab:red", label="Mismatch")
    else:
        ax.plot(xs, hr_arr, linestyle="None", marker="x", label="Hit rate")
    ax.plot(xs, fa_arr, linestyle="None", marker="o", label="False alarm")
    ax.set_ylim(0, 1); ax.set_xticks(xs); ax.set_xticklabels(session_labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("P(Lick)")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)

    ax_d = ax.twinx()
    if dual_stim:
        ax_d.plot(xs, dp_arr, linestyle="--", marker="o", color="green", label="% successful trials")
        ax_d.set_ylabel("% successful trials"); ax_d.set_ylim(0, 1); ax_d.grid(False)
    else:
        ax_d.plot(xs, dp_arr, linestyle="--", marker="o", color="green", label="d′")
        ax_d.set_ylabel("d′"); ax_d.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_d.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

    if created_fig:
        plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)

    if dual_stim:
        return xs, hr_arr, fa_arr, dp_arr, mismatch_arr, hr_right_arr, hr_left_arr
    return xs, hr_arr, fa_arr, dp_arr


# Plots compacts (une session / multi-souris)
def plot_single_session_rates(
    perf: Optional[np.ndarray],
    title: str = "HR vs FA (session)",
    zero_when_empty: bool = True,
    ax: Optional[plt.Axes] = None,
    save_name: Optional[str] = None,
    dual_stim: bool = False,
):
    """Plot compact pour UNE session (HR & FA, + HR par cote et Mismatch si dual_stim) + d′ en légende."""
    if dual_stim:
        hr, hr_right, hr_left, mismatch, fa, dprime = session_rates_dual(perf, zero_when_empty=zero_when_empty)
    else:
        hr, fa, dprime = session_rates(perf, zero_when_empty=zero_when_empty)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True

    if dual_stim:
        xs = np.array([1, 2, 3, 4], dtype=float)
        ax.plot([xs[0]], [hr_right], linestyle="None", marker="x", color="tab:blue", label="HR right")
        ax.plot([xs[1]], [hr_left], linestyle="None", marker="x", color="tab:orange", label="HR left")
        ax.plot([xs[2]], [mismatch], linestyle="None", marker="^", color="tab:red", label="Mismatch")
        ax.plot([xs[3]], [fa], linestyle="None", marker="o", label="FA")
        ax.set_xticks(xs); ax.set_xticklabels(["HR right", "HR left", "Mismatch", "FA"])
    else:
        xs = np.array([1, 2], dtype=float)
        ax.plot([xs[0]], [hr], linestyle="None", marker="x", label="HR")
        ax.plot([xs[1]], [fa], linestyle="None", marker="o", label="FA")
        ax.set_xticks(xs); ax.set_xticklabels(["HR", "FA"])

    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Lick)"); ax.set_title(title)

    dprime_txt = f"d′ = {dprime:.2f}" if not np.isnan(dprime) else "d′ : n/a"
    leg = ax.legend(loc="upper right", frameon=True, title=dprime_txt)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_alpha(0.9)
    ax.grid(True, axis="y", alpha=0.3)

    if created_fig:
        plt.tight_layout(); _save_fig(fig, category="session_summary", name=save_name or title)

    if dual_stim:
        return float(hr), float(hr_right), float(hr_left), float(mismatch), float(fa), float(dprime)
    return float(hr), float(fa), float(dprime)


# ancien modele, non utilise
# def plot_multi_mouse_single_session(
#     performance_list: Iterable[Optional[np.ndarray]],
#     mouse_labels: List[str],
#     title: str = "HR vs FA per mouse (single session)",
#     zero_when_empty: bool = True,
#     jitter: float = 0.06,
#     annotate_dprime: bool = True,
#     ax: Optional[plt.Axes] = None,
#     show: bool = True,
#     save_name: Optional[str] = None
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Superpose HR (x) et FA (o) pour plusieurs souris sur UNE session.
#     X = ["HR","FA"], Y = P(Lick)∈[0,1]. d′ annoté près de HR.
#     """
#     perfs = list(performance_list)
#     if len(perfs) != len(mouse_labels):
#         raise ValueError("mouse_labels doit avoir la même longueur que performance_list.")
#
#     n = len(perfs)
#     hr_arr = np.empty(n, dtype=float)
#     fa_arr = np.empty(n, dtype=float)
#     dp_arr = np.empty(n, dtype=float)
#
#     created_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(7.5, 5))
#         created_fig = True
#
#     x_hr, x_fa = 1.0, 2.0
#     offs = np.linspace(-jitter, jitter, n) if n > 1 else np.array([0.0])
#
#     for i, (perf, label) in enumerate(zip(perfs, mouse_labels)):
#         hr, fa, dp = session_rates(perf, zero_when_empty=zero_when_empty)
#         hr_arr[i], fa_arr[i], dp_arr[i] = hr, fa, dp
#
#         (line_hr,) = ax.plot([x_hr + offs[i]], [hr], linestyle="None", marker="x",
#                              label=f"{label} — d′={dp:.2f}")
#         color = line_hr.get_color()
#         ax.plot([x_fa + offs[i]], [fa], linestyle="None", marker="o",
#                 color=color, label="_nolegend_")
#
#         if annotate_dprime:
#             ax.annotate(f"{dp:.2f}",
#                         xy=(x_hr + offs[i], hr),
#                         xytext=(x_hr + offs[i] + 0.03, min(1.0, hr + 0.06)),
#                         fontsize=9, color=color)
#
#     ax.set_ylim(0, 1)
#     ax.set_xticks([x_hr, x_fa]); ax.set_xticklabels(["HR", "FA"])
#     ax.set_ylabel("P(Lick)"); ax.set_title(title)
#     ax.grid(True, axis="y", alpha=0.3)
#     ax.legend(ncols=1, frameon=True)
#
#     if created_fig and show:
#         plt.tight_layout(); _save_fig(fig, category="population", name=save_name or title)
#     return hr_arr, fa_arr, dp_arr
#
#
# def overlay_group_stats_on_current_axes(hr_arr: np.ndarray,
#                                         fa_arr: np.ndarray,
#                                         dp_arr: np.ndarray,
#                                         label: str = "Mean ± SD",
#                                         session_name: Optional[str] = None,
#                                         update_title: bool = True,
#                                         show_errorbars: bool = True) -> None:
#     """
#     Superpose sur la figure courante:
#       - HR_mean ± SD et FA_mean ± SD (optionnel)
#       - d′ moyen annoté
#       - Encadré avec μ, σ, σ² pour HR & FA
#     À appeler juste après plot_multi_mouse_single_session(...).
#     """
#     ax = plt.gca()
#     x_hr, x_fa = 1.0, 2.0
#
#     hr_mean = float(np.nanmean(hr_arr)) if hr_arr.size else np.nan
#     fa_mean = float(np.nanmean(fa_arr)) if fa_arr.size else np.nan
#     dp_mean = float(np.nanmean(dp_arr)) if dp_arr.size else np.nan
#
#     hr_sd = float(np.nanstd(hr_arr, ddof=1)) if hr_arr.size > 1 else 0.0
#     fa_sd = float(np.nanstd(fa_arr, ddof=1)) if fa_arr.size > 1 else 0.0
#
#     hr_var = float(np.nanvar(hr_arr, ddof=1)) if hr_arr.size > 1 else 0.0
#     fa_var = float(np.nanvar(fa_arr, ddof=1)) if fa_arr.size > 1 else 0.0
#
#     if show_errorbars:
#         ax.errorbar([x_hr], [hr_mean], yerr=[hr_sd], fmt='s', ms=8, capsize=4, label=f"{label} HR")
#         ax.errorbar([x_fa], [fa_mean], yerr=[fa_sd], fmt='s', ms=8, capsize=4, label=f"{label} FA")
#
#     if np.isfinite(dp_mean):
#         ax.annotate(f"mean d′ = {dp_mean:.2f}",
#                     xy=(x_hr, hr_mean),
#                     xytext=(x_hr + 0.08, min(1.0, hr_mean + 0.10)),
#                     fontsize=10,
#                     fontweight="bold",
#                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
#
#     stats_lines = [
#         f"HR: μ={hr_mean:.2f}, σ={hr_sd:.2f}, σ²={hr_var:.3f}",
#         f"FA: μ={fa_mean:.2f}, σ={fa_sd:.2f}, σ²={fa_var:.3f}",
#     ]
#     if np.isfinite(dp_mean):
#         stats_lines.append(f"d′ mean: {dp_mean:.2f}")
#
#     ax.text(0.02, 0.98, "\n".join(stats_lines),
#             transform=ax.transAxes, va="top", ha="left",
#             fontsize=10,
#             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="gray", alpha=0.9))
#
#     ax.set_ylim(0, 1)
#     ax.grid(True, axis="y", alpha=0.3)
#     ax.legend(frameon=True)
#
#     if session_name and update_title:
#         old_title = ax.get_title()
#         if old_title and session_name not in old_title:
#             ax.set_title(f"{old_title}\nSession: {session_name}")
#         elif not old_title:
#             ax.set_title(f"Session: {session_name}")
#

# RPE par lick (scatter)
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


def session_rates_wa(perf: Optional[np.ndarray], zero_when_empty: bool = True) -> Tuple[float, float, float]:
    """Calcule (HR_stim1, HR_stim2, FA) sur toute la session. perf: colonnes
    [t, stim_code(0/1/2), lick_detected, latency, reward_detected, outcome]."""
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

    return _rate(s1_mask, 1), _rate(s2_mask, 1), _rate(catch_mask, 3)


def plot_wdt_block_rates_wa(performance: np.ndarray, wdt_params, max_trials: int = 400,
                            title: str = "", save_name: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HR(stim1)/HR(stim2)/FA par blocs a l'interieur d'une session."""
    n = min(len(performance), max_trials)
    if n == 0:
        raise ValueError("performance est vide.")
    block = max(1, int(getattr(wdt_params, "block_numb", 5)))

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
    ax.plot(xs, hr2s, linestyle="None", marker="^", label="Hit rate stim2 (auditory)")
    ax.plot(xs, fas, linestyle="None", marker="o", label="False alarm (catch)")
    ax.set_xlim(0, max_trials); ax.set_ylim(0, 1)
    ax.set_xlabel("Trial #"); ax.set_ylabel("P(Lick)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); _save_fig(fig, category="block_rates", name=save_name or title or "block_rates")
    return np.array(xs), np.array(hr1s), np.array(hr2s), np.array(fas)


def plot_session_rates_wa(performance_list: Iterable[Optional[np.ndarray]], session_labels: List[str],
                          title: str = "HR(stim1/stim2) & FA(catch) across sessions",
                          zero_when_empty: bool = True, save_name: Optional[str] = None,
                          switch_session: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X = sessions ; Y = P(Lick) pour HR(stim1), HR(stim2), FA(catch). Ligne verticale au
    switch de contingence si `switch_session` est fourni."""
    perfs = list(performance_list)
    if len(session_labels) != len(perfs):
        raise ValueError("session_labels doit avoir la meme longueur que performance_list.")

    rates = [session_rates_wa(p, zero_when_empty=zero_when_empty) for p in perfs]
    hr1_arr = np.array([r[0] for r in rates], dtype=float)
    hr2_arr = np.array([r[1] for r in rates], dtype=float)
    fa_arr = np.array([r[2] for r in rates], dtype=float)
    xs = np.arange(1, len(perfs) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, hr1_arr, linestyle="-", marker="x", color="tab:blue", label="Hit rate stim1 (whisker)")
    ax.plot(xs, hr2_arr, linestyle="-", marker="^", color="tab:orange", label="Hit rate stim2 (auditory)")
    ax.plot(xs, fa_arr, linestyle="-", marker="o", color="tab:green", label="False alarm (catch)")
    if switch_session is not None and 1 <= switch_session <= len(xs):
        ax.axvline(switch_session - 0.5, linestyle="--", color="gray", alpha=0.6, label="contingency switch")
    ax.set_ylim(0, 1); ax.set_xticks(xs); ax.set_xticklabels(session_labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("P(Lick)")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)
    return xs, hr1_arr, hr2_arr, fa_arr


def plot_single_session_rates_wa(perf: Optional[np.ndarray], title: str = "HR(stim1/stim2) vs FA (session)",
                                 zero_when_empty: bool = True, save_name: Optional[str] = None
                                ) -> Tuple[float, float, float]:
    """Plot compact pour UNE session (HR stim1, HR stim2, FA)."""
    hr1, hr2, fa = session_rates_wa(perf, zero_when_empty=zero_when_empty)

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


def plot_stim_gains(wdt_bundle, config, title: str = "Stimulus gain evolution",
                    save_name: Optional[str] = None) -> None:
    """
    Pour chaque session, trace le gain de stimulus de stim1 et stim2 (voir
    Mouse.exp_update_stim1_wa/2_wa) — le meme gain monte a la recompense et descend au
    mismatch de contingence. S'effondre pour le stimulus qui cesse d'etre recompense apres
    le switch, et grimpe pour celui qui prend le relais.
    """
    labels = wdt_bundle.labels
    xs = np.arange(1, len(labels) + 1)
    g1 = np.array([r.stim1_gain for r in wdt_bundle.results])
    g2 = np.array([r.stim2_gain for r in wdt_bundle.results])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, g1, linestyle="-", marker="x", color="tab:blue", linewidth=2, label="Gain stim1 (whisker)")
    ax.plot(xs, g2, linestyle="-", marker="^", color="tab:orange", linewidth=2, label="Gain stim2 (auditory)")

    ax.axvline(config.WA_SWITCH_SESSION - 0.5, linestyle="--", color="gray", alpha=0.6, label="contingency switch")
    ax.axhline(0.0, linestyle="-", color="black", alpha=0.3, linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("Sessions"); ax.set_ylabel("Gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(); _save_fig(fig, category="learning_curve", name=save_name or title)


# Fichier texte recapitulatif des parametres numeriques de la simulation
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
