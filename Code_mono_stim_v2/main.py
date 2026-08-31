# ============================ SESSION FLOW (2 FL + 10 WDT + PSYCHO) ============================
import os
from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# ------------------------------ SAUVEGARDE DES FIGURES (au lieu des pop-ups) ------------------------------
SAVE_FIGS: bool = True   # True => les figures sont enregistrées dans FIG_DIR, aucune fenêtre ne s'ouvre
FIG_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)
    matplotlib.use("Agg")  # backend non-interactif : pas de fenêtre popup

    _fig_counter = {"n": 0}

    def _figure_title(fig) -> str:
        if fig._suptitle is not None and fig._suptitle.get_text():
            return fig._suptitle.get_text()
        for ax in fig.axes:
            if ax.get_title():
                return ax.get_title()
        return f"figure_{_fig_counter['n']:03d}"

    def _save_instead_of_show(*args, **kwargs):
        _fig_counter["n"] += 1
        fig = plt.gcf()
        title = _figure_title(fig)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()[:80]
        fname = f"{_fig_counter['n']:03d}_{safe_title or 'figure'}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=130, bbox_inches="tight")
        plt.close(fig)

    plt.show = _save_instead_of_show
    print(f"[SAVE_FIGS] Les figures seront enregistrées dans: {FIG_DIR}")

from models import (
    Mouse,
    SessionBaseInfo,
    FreeLickingSessionParams,
    FreeLickingSessionState,
    MouseSessionState,
    WDTSesssionParams,
    WDTSesssionState,
)

from functions import (
    function_fl_session,
    function_mouse_lick,
    function_update_mouse_state,
    function_wdt_session,
    function_performance_wdt,
    new_rpe_buffer,
    online_sigmoid_update,
    extract_rpe_per_lick,
    population_hr_fa_plot,
    sample_trial_value_cost,
)


from plotting import (
    plot_traces,
    plot_wdt_block_rates,
    plot_session_rates,
    plot_single_session_rates,
    plot_rpe_per_lick,
    plot_noise_gain_sigmoid,
    plot_abs_rpe_per_lick,
)

# ------------------------------ SWITCHES POUR LES PLOTS ------------------------------
PLOT_TRACES: bool = True                       # Traces temporelles (FL1, FL2, WDT1..10, TEST)
PLOT_BLOCK_HRFA: bool = True                   # HR/FA par blocs pour toutes les WDT
MAX_TRIALS_BLOCKS: int = 400                    # borne X pour la fenêtre de blocs
PLOT_SESSIONS_COMPARISON: bool = True          # HR, FA, d′ sur une figure pour WDT1..10
PLOT_SINGLE_SESSION_ALL: bool = True           # Petit plot HR vs FA (+ d′) pour chaque WDT
PLOT_STOCHASTIC_IN_POPULATION : bool = True
PLOT_RPE_ALL: bool = True                      # RPE par lick pour FL1/2 et WDT1..10
PLOT_ABS_RPE_ALL: bool = True
PLOT_PSYCHO_TEST: bool = True                   # Courbe psycho pour WDT_TEST (multi-amps)


# ------------------------------ Réglages généraux ------------------------------
NUM_WDT = 10                                # nombre de sessions WDT (ici 10)
DEFAULT_TYPES = [0.0, 1.0]                  # WDT binaires
WDT_TEST_TYPES = [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5]  # psycho multi-amp

"""SEED = 80
random.seed(SEED)   # stdlib
np.random.seed(SEED)  # NumPy"""

# ------------------------------ Noise modulation (globale) ------------------------------
# DÉSACTIVÉ : ce mécanisme hérité modulait mouse.noise[2] (le gain du générateur Nd) via sa
# propre sigmoïde sur une moyenne glissante de |RPE| — redondant avec, et incohérent avec, la
# vraie fonction Uncertainty (U) qui pilote maintenant le poids exploitation/exploration
# directement dans D. Nd doit rester un générateur à gain CONSTANT ; c'est le poids qui varie
# avec U, pas l'amplitude du bruit lui-même. Remis à False pour éviter le double comptage.
USE_SIGMOID_NOISE: bool = False    # on/off pour toutes les sessions
SIG_GAIN_MIN: float   = 0.08       # gmin (borne basse du gain)
SIG_GAIN_MAX: float   = 0.28       # gmax (borne haute du gain)
SIG_X0: float         = 0.4        # abscisse du point d'inflexion
SIG_SLOPE: float      = 14.0       # pente de la sigmoïde
SIG_AVG_LICKS: int    = 15         # taille de la fenêtre (nb de licks)
SIG_UPDATE_EVERY: int = 1          # mise à jour du gain toutes les P licks
PLOT_SIGMOID_MAPPING: bool = False # pour visualiser g(m) avec les params actuels


# ------------------------------ Internal variance ------------------------------
N_MICE = 10
NOISE_GAIN = 0.06
LEARNING_STIM = 0.02
SESSION_NAME = "WDT10"

# ------------------------------ Value / Cost dynamiques (fusion du travail du collègue) ------------------------------
# En Free Licking, V/C restent statiques (mouse.value=1.0, mouse.cost=0.0, cf. models.py).
# En WDT, si VC_VARY=True, on tire un V (taille de goutte) et un C (distance/difficulté du
# spout) à chaque nouveau trial, moyennés sur une fenêtre glissante (comme dans
# Code_mono_stim/main.py — sample_trial_value_cost).
VC_VARY: bool = True
V_RANGE: tuple = (0.5, 1.0)         # taille de la goutte (droplet size), tirée uniformément
C_RANGE: tuple = (0.0, 0.5)         # distance/difficulté du spout, tirée uniformément
VC_BUFFER_SIZE: int = 5             # nb de derniers trials moyennés pour expected_V / expected_C


# =============================================================================================
#                                   INITIALISATION GLOBALE
# =============================================================================================
session_info = SessionBaseInfo()
mouse = Mouse()
LICK_THRS_FL_USED = mouse.lick_thrs  # capturé avant que le seuil ne soit basculé pour les WDT (voir plus bas)

noise_trace_fl1 = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
noise_trace_fl2 = None

wdt_noise_traces: list[np.ndarray] = []
wdt_sessions:      list[WDTSesssionState] = []
wdt_mice:          list[MouseSessionState] = []
wdt_perfs:         list[np.ndarray] = []
wdt_labels:        list[str] = [f"WDT{i}" for i in range(1, NUM_WDT + 1)]

# TEST (psycho multi-amp)
noise_trace_wdt_test = None
log_session_wdt_test = None
log_mouse_wdt_test   = None
log_performance_wdt_test = None

# =============================================================================================
#                                        FL — SESSION 1
# =============================================================================================
fl_params = FreeLickingSessionParams()
fl_session = FreeLickingSessionState(); fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
mouse_session = MouseSessionState();    mouse_session.initialize(session_info.number_bin, mouse.motivation[0])

buf_fl1, cnt_fl1 = new_rpe_buffer(SIG_AVG_LICKS)
noise_trace_fl1[0, 0] = float(mouse.noise[2])

for i in range(1, session_info.number_bin):
    reward_t, stim_t, fl_session = function_fl_session(
        fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
    )
    _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
    mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

    cnt_fl1 = online_sigmoid_update(
        mouse, mouse_session, i,
        buf_fl1, cnt_fl1,
        enabled=USE_SIGMOID_NOISE,
        m0=SIG_X0, k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN, gmax=SIG_GAIN_MAX,
        update_every=SIG_UPDATE_EVERY,
    )
    noise_trace_fl1[i, 0] = float(mouse.noise[2])

log_session_fl1 = fl_session
log_mouse_fl1   = mouse_session

# =============================================================================================
#                                        FL — SESSION 2
# =============================================================================================
fl_params.forced_reward = (0, 0)

lick_sum   = float(np.sum(log_mouse_fl1.lick))
reward_sum = float(np.sum(log_session_fl1.reward))
init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
init_uncert = float(log_mouse_fl1.uncertainty[-1, 0])  # U persiste entre sessions (pas de forgetting)

fl_session   = FreeLickingSessionState(); fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
mouse_session = MouseSessionState();      mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert)

buf_fl2, cnt_fl2 = new_rpe_buffer(SIG_AVG_LICKS)
noise_trace_fl2  = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
noise_trace_fl2[0, 0] = float(mouse.noise[2])

for i in range(1, session_info.number_bin):
    reward_t, stim_t, fl_session = function_fl_session(
        fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
    )
    _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
    mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

    cnt_fl2 = online_sigmoid_update(
        mouse, mouse_session, i,
        buf_fl2, cnt_fl2,
        enabled=USE_SIGMOID_NOISE,
        m0=SIG_X0, k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN, gmax=SIG_GAIN_MAX,
        update_every=SIG_UPDATE_EVERY,
    )
    noise_trace_fl2[i, 0] = float(mouse.noise[2])

log_session_fl2 = fl_session
log_mouse_fl2   = mouse_session

# =============================================================================================
#                                   WDT — SESSIONS 1..10
# =============================================================================================
prev_ms = log_mouse_fl2
prev_rw = log_session_fl2.reward

# Bascule sur le seuil WDT (échelle différente une fois V/C dynamiques, cf. models.py)
mouse.lick_thrs = mouse.lick_thrs_wdt

for s_idx in range(1, NUM_WDT + 1):
    wdt_params = WDTSesssionParams()
    wdt_params.trial_types = DEFAULT_TYPES[:]  # binaire (0 vs 1)

    lick_sum   = float(np.sum(prev_ms.lick))
    reward_sum = float(np.sum(prev_rw))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
    init_uncert = float(prev_ms.uncertainty[-1, 0])  # U persiste entre sessions (pas de forgetting)

    wdt_session = WDTSesssionState(); wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState(); mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert)

    # trace du noise gain pour cette session
    noise_trace_wdt = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
    noise_trace_wdt[0, 0] = float(mouse.noise[2])

    buf_wdt, cnt_wdt = new_rpe_buffer(SIG_AVG_LICKS)
    buf_v, buf_c = deque(maxlen=VC_BUFFER_SIZE), deque(maxlen=VC_BUFFER_SIZE)

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, wdt_session, new_trial = function_wdt_session(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        if VC_VARY and new_trial:
            sample_trial_value_cost(mouse, buf_v, buf_c, V_RANGE, C_RANGE)
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

        cnt_wdt = online_sigmoid_update(
            mouse, mouse_session, i,
            buf_wdt, cnt_wdt,
            enabled=USE_SIGMOID_NOISE,
            m0=SIG_X0, k=SIG_SLOPE,
            gmin=SIG_GAIN_MIN, gmax=SIG_GAIN_MAX,
            update_every=SIG_UPDATE_EVERY,
        )
        noise_trace_wdt[i, 0] = float(mouse.noise[2])

    # perf & logs
    perf = function_performance_wdt(
        wdt_params=wdt_params,
        session_info=session_info,
        wdt_session=wdt_session,
        mouse_session=mouse_session,
    )

    if perf.size > 0:
        stim_mask  = perf[:, 1] > 0
        catch_mask = ~stim_mask
        hr = (np.sum((perf[:, 5] == 1) & stim_mask)  / np.sum(stim_mask))  if np.sum(stim_mask)  > 0 else 0.0
        fa = (np.sum((perf[:, 5] == 3) & catch_mask) / np.sum(catch_mask)) if np.sum(catch_mask) > 0 else 0.0
    else:
        hr, fa = 0.0, 0.0

    print(f"Hit Rate WDT{s_idx} : {hr:.2f}")
    print(f"FA  Rate WDT{s_idx} : {fa:.2f}")

    # stocker
    wdt_perfs.append(perf)
    wdt_sessions.append(wdt_session)
    wdt_mice.append(mouse_session)
    wdt_noise_traces.append(noise_trace_wdt)

    prev_ms = mouse_session
    prev_rw = wdt_session.reward

# =============================================================================================
#                         WDT_TEST — PSYCHOMÉTRIQUE (multi-amp)
# =============================================================================================
wdt_test_params = WDTSesssionParams()
wdt_test_params.trial_types = WDT_TEST_TYPES[:]

lick_sum   = float(np.sum(prev_ms.lick))
reward_sum = float(np.sum(prev_rw))
init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
init_uncert = float(prev_ms.uncertainty[-1, 0])  # U persiste entre sessions (pas de forgetting)

wdt_test_session = WDTSesssionState(); wdt_test_session.initialize(session_info.number_bin, wdt_test_params.no_lick_wind, wdt_test_params.iti)
wdt_test_mouse   = MouseSessionState(); wdt_test_mouse.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert)

noise_trace_wdt_test = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
noise_trace_wdt_test[0, 0] = float(mouse.noise[2])

buf_wdtT, cnt_wdtT = new_rpe_buffer(SIG_AVG_LICKS)
buf_v_test, buf_c_test = deque(maxlen=VC_BUFFER_SIZE), deque(maxlen=VC_BUFFER_SIZE)

for i in range(1, session_info.number_bin):
    reward_t, stim_t, wdt_test_session, new_trial = function_wdt_session(
        wdt_test_params, wdt_test_session, session_info, int(wdt_test_mouse.lick[i - 1, 0]), i
    )
    if VC_VARY and new_trial:
        sample_trial_value_cost(mouse, buf_v_test, buf_c_test, V_RANGE, C_RANGE)
    _, wdt_test_mouse = function_mouse_lick(mouse, wdt_test_mouse, i)
    mouse, wdt_test_mouse = function_update_mouse_state(mouse, wdt_test_mouse, session_info, i, stim_t, reward_t)

    cnt_wdtT = online_sigmoid_update(
        mouse, wdt_test_mouse, i,
        buf_wdtT, cnt_wdtT,
        enabled=USE_SIGMOID_NOISE,
        m0=SIG_X0, k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN, gmax=SIG_GAIN_MAX,
        update_every=SIG_UPDATE_EVERY,
    )
    noise_trace_wdt_test[i, 0] = float(mouse.noise[2])

log_session_wdt_test = wdt_test_session
log_mouse_wdt_test   = wdt_test_mouse

log_performance_wdt_test = function_performance_wdt(
    wdt_params=wdt_test_params,
    session_info=session_info,
    wdt_session=log_session_wdt_test,
    mouse_session=log_mouse_wdt_test
)

# =========================== PLOTTING ============================

if PLOT_TRACES:
    plot_traces(session_info.time_vector, log_mouse_fl1, log_session_fl1.reward,
                stim_array=None, title="Free Licking — Session 1", noise_trace=noise_trace_fl1,
                lick_threshold=LICK_THRS_FL_USED)
    plot_traces(session_info.time_vector, log_mouse_fl2, log_session_fl2.reward,
                stim_array=None, title="Free Licking — Session 2", noise_trace=noise_trace_fl2,
                lick_threshold=LICK_THRS_FL_USED)
    for k, lbl in enumerate(wdt_labels):
        plot_traces(session_info.time_vector, wdt_mice[k], wdt_sessions[k].reward,
                    stim_array=wdt_sessions[k].stim1, title=f"Whisker Detection Task — {lbl}",
                    noise_trace=wdt_noise_traces[k], lick_threshold=mouse.lick_thrs)
    plot_traces(session_info.time_vector, log_mouse_wdt_test, log_session_wdt_test.reward,
                stim_array=log_session_wdt_test.stim1, title="WDT Test (multi-amp)",
                noise_trace=noise_trace_wdt_test, lick_threshold=mouse.lick_thrs)

if PLOT_BLOCK_HRFA:
    _wdt_params = WDTSesssionParams()
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_wdt_block_rates(perf, _wdt_params, max_trials=MAX_TRIALS_BLOCKS,
                             title=f"{lbl} — HR/FA par blocs")

if PLOT_SESSIONS_COMPARISON:
    plot_session_rates(wdt_perfs, wdt_labels, title="Apprentissage — HR, FA & d′ (WDT1..WDT10)")

if PLOT_SINGLE_SESSION_ALL:
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_single_session_rates(perf, title=f"{lbl} — HR vs FA")

if PLOT_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl1, title="FL1 — RPE par lick")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl2, title="FL2 — RPE par lick")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, wdt_log.stim1, session_info)
        plot_rpe_per_lick(tbl, title=f"{lbl} — RPE par lick")
        
if PLOT_ABS_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl1, title="FL1 — |RPE| par lick")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl2, title="FL2 — |RPE| par lick")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, wdt_log.stim1, session_info)
        plot_abs_rpe_per_lick(tbl, title=f"{lbl} — |RPE| par lick")


if PLOT_PSYCHO_TEST and (log_performance_wdt_test is not None and log_performance_wdt_test.size):
    perf = log_performance_wdt_test
    stim_amps = perf[:, 1]
    outcomes  = perf[:, 5]
    unique_stims = np.unique(stim_amps)
    hit_rates = []
    for s in unique_stims:
        m = stim_amps == s
        hit_rates.append(np.sum(outcomes[m] == 1) / np.sum(m) if np.sum(m) > 0 else np.nan)
    plt.figure(figsize=(8, 5))
    plt.plot(unique_stims, hit_rates, 'o-', label='Hit rate')
    plt.xlabel('Stimulus amplitude'); plt.ylabel('Hit rate')
    plt.title('Psychometric curve — WDT Test (multi-amp)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


if PLOT_SIGMOID_MAPPING:
    plot_noise_gain_sigmoid(
        m0=SIG_X0,
        k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN,
        gmax=SIG_GAIN_MAX,
        title="Mapping g(m) pour le noise gain"
    )
    
    
    
if PLOT_STOCHASTIC_IN_POPULATION :
    hr_arr, fa_arr, dp_arr = population_hr_fa_plot(
    n_mice=N_MICE,
    noise_gain=NOISE_GAIN,
    learning_stim=LEARNING_STIM,
    session_name=SESSION_NAME,
)
