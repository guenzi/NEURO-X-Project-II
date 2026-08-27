# ============================ SESSION FLOW (2 FL + 10 WDT + PSYCHO) ============================
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

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
    plot_value_cost_trials,
    save_parameters_txt,
    init_results_dir,
    _save_fig,
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
SAVE_PARAMETERS_TXT: bool = True                # Fichier texte recapitulatif de tous les parametres numeriques


# ------------------------------ Réglages généraux ------------------------------
NUM_WDT = 10                                # nombre de sessions WDT (ici 10)
DEFAULT_TYPES = [0.0, 1.0]                  # WDT binaires
WDT_TEST_TYPES = [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5]  # psycho multi-amp

"""SEED = 80
random.seed(SEED)   # stdlib
np.random.seed(SEED)  # NumPy"""

# ------------------------------ Noise modulation (globale) ------------------------------
USE_SIGMOID_NOISE: bool = True     # on/off pour toutes les sessions
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

# ------------------------------ Value / Cost (decision gate: V*M - C) ------------------------------
REWARD_VALUE = 1.0   # V — laisser a 1.0 pour ne rien changer au comportement actuel
COST = 0.2           # C — laisser a 0.0 pour ne rien changer au comportement actuel

VC_VARY: bool = True        # si True: V/C tires aleatoirement a chaque trial WDT (goutte + distance)
V_RANGE = (0.5, 1.0)         # taille de la goutte (droplet size), tiree uniformement
C_RANGE = (0.0, 0.5)         # distance/difficulte du spout, tiree uniformement
VC_BUFFER_SIZE = 5           # nb de derniers trials moyennes pour expected_V / expected_C

# Seuil de lechage — separe par type de session car l'echelle de decision_gate (V*M-C)
# differe entre Free Licking (V/C constants) et WDT (V/C variables si VC_VARY=True).
# Les deux valent 1.0 par defaut: avec V=1, C=0, le comportement est strictement identique a avant.
LICK_THRS_FL = 1.0
if VC_VARY:
    LICK_THRS_WDT = 0.45
else:
    LICK_THRS_WDT = 1.0

_results_suffix = "VCvary" if VC_VARY else f"V{REWARD_VALUE:g}_C{COST:g}"
init_results_dir(suffix=_results_suffix)


# =============================================================================================
#                                   INITIALISATION GLOBALE
# =============================================================================================
session_info = SessionBaseInfo()
mouse = Mouse(reward_value=REWARD_VALUE, cost=COST, lick_thrs=LICK_THRS_FL, lick_thrs_wdt=LICK_THRS_WDT)

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

fl_session   = FreeLickingSessionState(); fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
mouse_session = MouseSessionState();      mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect)

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

wdt_vc_logs: list[dict] = []  # un dict par session WDT: trial_times, v, c, expected_v, expected_c

for s_idx in range(1, NUM_WDT + 1):
    wdt_params = WDTSesssionParams()
    wdt_params.trial_types = DEFAULT_TYPES[:]  # binaire (0 vs 1)

    lick_sum   = float(np.sum(prev_ms.lick))
    reward_sum = float(np.sum(prev_rw))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0

    wdt_session = WDTSesssionState(); wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState(); mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect)

    # trace du noise gain pour cette session
    noise_trace_wdt = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
    noise_trace_wdt[0, 0] = float(mouse.noise[2])

    buf_wdt, cnt_wdt = new_rpe_buffer(SIG_AVG_LICKS)
    buf_v, buf_c = deque(maxlen=VC_BUFFER_SIZE), deque(maxlen=VC_BUFFER_SIZE)
    vc_trial_times: list[float] = []
    vc_v: list[float] = []
    vc_c: list[float] = []
    vc_expected_v: list[float] = []
    vc_expected_c: list[float] = []

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, wdt_session, new_trial = function_wdt_session(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        if VC_VARY and new_trial:
            v_trial, c_trial, ev, ec = sample_trial_value_cost(mouse, buf_v, buf_c, V_RANGE, C_RANGE)
            vc_trial_times.append(i * session_info.resolution)
            vc_v.append(v_trial)
            vc_c.append(c_trial)
            vc_expected_v.append(ev)
            vc_expected_c.append(ec)
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i, threshold=mouse.lick_thrs_wdt)
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
    wdt_vc_logs.append({
        "trial_times": vc_trial_times, "v": vc_v, "c": vc_c,
        "expected_v": vc_expected_v, "expected_c": vc_expected_c,
    })

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

wdt_test_session = WDTSesssionState(); wdt_test_session.initialize(session_info.number_bin, wdt_test_params.no_lick_wind, wdt_test_params.iti)
wdt_test_mouse   = MouseSessionState(); wdt_test_mouse.initialize(session_info.number_bin, mouse.motivation[0], init_expect)

noise_trace_wdt_test = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
noise_trace_wdt_test[0, 0] = float(mouse.noise[2])

buf_wdtT, cnt_wdtT = new_rpe_buffer(SIG_AVG_LICKS)
buf_v_test, buf_c_test = deque(maxlen=VC_BUFFER_SIZE), deque(maxlen=VC_BUFFER_SIZE)
vc_trial_times_test: list[float] = []
vc_v_test: list[float] = []
vc_c_test: list[float] = []
vc_expected_v_test: list[float] = []
vc_expected_c_test: list[float] = []

for i in range(1, session_info.number_bin):
    reward_t, stim_t, wdt_test_session, new_trial = function_wdt_session(
        wdt_test_params, wdt_test_session, session_info, int(wdt_test_mouse.lick[i - 1, 0]), i
    )
    if VC_VARY and new_trial:
        v_trial, c_trial, ev, ec = sample_trial_value_cost(mouse, buf_v_test, buf_c_test, V_RANGE, C_RANGE)
        vc_trial_times_test.append(i * session_info.resolution)
        vc_v_test.append(v_trial)
        vc_c_test.append(c_trial)
        vc_expected_v_test.append(ev)
        vc_expected_c_test.append(ec)
    _, wdt_test_mouse = function_mouse_lick(mouse, wdt_test_mouse, i, threshold=mouse.lick_thrs_wdt)
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
                save_name="free_licking_1", threshold=mouse.lick_thrs)
    plot_traces(session_info.time_vector, log_mouse_fl2, log_session_fl2.reward,
                stim_array=None, title="Free Licking — Session 2", noise_trace=noise_trace_fl2,
                save_name="free_licking_2", threshold=mouse.lick_thrs)
    for k, lbl in enumerate(wdt_labels):
        plot_traces(session_info.time_vector, wdt_mice[k], wdt_sessions[k].reward,
                    stim_array=wdt_sessions[k].stim1, title=f"Whisker Detection Task — {lbl}",
                    noise_trace=wdt_noise_traces[k], save_name=lbl.lower(), threshold=mouse.lick_thrs_wdt)
    plot_traces(session_info.time_vector, log_mouse_wdt_test, log_session_wdt_test.reward,
                stim_array=log_session_wdt_test.stim1, title="WDT Test (multi-amp)",
                noise_trace=noise_trace_wdt_test, save_name="wdt_test", threshold=mouse.lick_thrs_wdt)

    if VC_VARY:
        for lbl, vc in zip(wdt_labels, wdt_vc_logs):
            if vc["trial_times"]:
                plot_value_cost_trials(
                    vc["trial_times"], vc["v"], vc["c"], vc["expected_v"], vc["expected_c"],
                    title=f"Whisker Detection Task — {lbl} — Value/Cost",
                    save_name=f"{lbl.lower()}_value_cost",
                )
        if vc_trial_times_test:
            plot_value_cost_trials(
                vc_trial_times_test, vc_v_test, vc_c_test, vc_expected_v_test, vc_expected_c_test,
                title="WDT Test (multi-amp) — Value/Cost",
                save_name="wdt_test_value_cost",
            )

if PLOT_BLOCK_HRFA:
    _wdt_params = WDTSesssionParams()
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_wdt_block_rates(perf, _wdt_params, max_trials=MAX_TRIALS_BLOCKS,
                             title=f"{lbl} — HR/FA by block", save_name=lbl.lower())

if PLOT_SESSIONS_COMPARISON:
    plot_session_rates(wdt_perfs, wdt_labels, title="Learning — HR, FA & d′ (WDT1..WDT10)",
                        save_name="hr_fa_dprime_all_sessions")

if PLOT_SINGLE_SESSION_ALL:
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_single_session_rates(perf, title=f"{lbl} — HR vs FA", save_name=lbl.lower())

if PLOT_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl1, title="FL1 — RPE per lick", save_name="free_licking_1_rpe")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl2, title="FL2 — RPE per lick", save_name="free_licking_2_rpe")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, wdt_log.stim1, session_info)
        plot_rpe_per_lick(tbl, title=f"{lbl} — RPE per lick", save_name=f"{lbl.lower()}_rpe")

if PLOT_ABS_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl1, title="FL1 — |RPE| per lick", save_name="free_licking_1_abs_rpe")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl2, title="FL2 — |RPE| per lick", save_name="free_licking_2_abs_rpe")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, wdt_log.stim1, session_info)
        plot_abs_rpe_per_lick(tbl, title=f"{lbl} — |RPE| per lick", save_name=f"{lbl.lower()}_abs_rpe")


if PLOT_PSYCHO_TEST and (log_performance_wdt_test is not None and log_performance_wdt_test.size):
    perf = log_performance_wdt_test
    stim_amps = perf[:, 1]
    outcomes  = perf[:, 5]
    unique_stims = np.unique(stim_amps)
    hit_rates = []
    for s in unique_stims:
        m = stim_amps == s
        hit_rates.append(np.sum(outcomes[m] == 1) / np.sum(m) if np.sum(m) > 0 else np.nan)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(unique_stims, hit_rates, 'o-', label='Hit rate')
    plt.xlabel('Stimulus amplitude'); plt.ylabel('Hit rate')
    plt.title('Psychometric curve — WDT Test (multi-amp)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    _save_fig(fig, category="psychometric", name="hit_rate_vs_stimulus_amplitude")


if PLOT_SIGMOID_MAPPING:
    plot_noise_gain_sigmoid(
        m0=SIG_X0,
        k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN,
        gmax=SIG_GAIN_MAX,
        title="Noise gain mapping g(m)",
        save_name="noise_gain_mapping"
    )
    
    
    
if PLOT_STOCHASTIC_IN_POPULATION :
    hr_arr, fa_arr, dp_arr = population_hr_fa_plot(
    n_mice=N_MICE,
    noise_gain=NOISE_GAIN,
    learning_stim=LEARNING_STIM,
    session_name=SESSION_NAME,
)

if SAVE_PARAMETERS_TXT:
    _ref_mouse = Mouse()
    _ref_session = SessionBaseInfo()
    _ref_fl = FreeLickingSessionParams()
    _ref_wdt = WDTSesssionParams()

    param_rows = [
        ("SESSION TIMING", None),
        ("Session duration (min)", _ref_session.duration),
        ("Time resolution (s)", _ref_session.resolution),
        ("Number of time bins", _ref_session.number_bin),

        ("MOUSE — NOISE & DECISION", None),
        ("Noise gamma shape", _ref_mouse.noise[0]),
        ("Noise gamma scale", _ref_mouse.noise[1]),
        ("Noise gain (initial)", _ref_mouse.noise[2]),
        ("Lick threshold — Free Licking", LICK_THRS_FL),
        ("Lick threshold — WDT", LICK_THRS_WDT),

        ("MOUSE — MOTIVATION", None),
        ("Motivation (initial)", _ref_mouse.motivation[0]),
        ("Motivation loss per reward", _ref_mouse.motivation[1]),

        ("MOUSE — VALUE & COST (DECISION GATE: V·M − C)", None),
        ("Reward value (V)", REWARD_VALUE),
        ("Cost (C)", COST),
        ("Per-trial random V/C (WDT only)", VC_VARY),
        ("Value range (droplet size)", f"{V_RANGE[0]}–{V_RANGE[1]}" if VC_VARY else "n/a"),
        ("Cost range (spout distance)", f"{C_RANGE[0]}–{C_RANGE[1]}" if VC_VARY else "n/a"),
        ("Expected V/C averaging window (trials)", VC_BUFFER_SIZE if VC_VARY else "n/a"),

        ("MOUSE — EXPECTATION UPDATE (REWARD)", None),
        ("Tau (reward, s)", _ref_mouse.exp_update_reward[0]),
        ("Gain (reward)", _ref_mouse.exp_update_reward[1]),

        ("MOUSE — EXPECTATION UPDATE (NO REWARD)", None),
        ("Tau (no reward, initial, s)", _ref_mouse.exp_update_no_reward[0]),
        ("Gain (no reward)", _ref_mouse.exp_update_no_reward[1]),

        ("MOUSE — EXPECTATION UPDATE (STIMULUS)", None),
        ("Tau (stimulus, s)", _ref_mouse.exp_update_stim[0]),
        ("Gain (stimulus, initial)", _ref_mouse.exp_update_stim[1]),

        ("MOUSE — LEARNING RULES", None),
        ("Consecutive non-rewarded licks threshold", _ref_mouse.learning_nonrew_lick[0]),
        ("Tau (no reward) increment", _ref_mouse.learning_nonrew_lick[1]),
        ("Stimulus gain learning rate", _ref_mouse.learning_stim),
        ("Eligibility trace tau (s)", _ref_mouse.tau_eligibility),

        ("NOISE GAIN MODULATION (SIGMOID)", None),
        ("Enabled", USE_SIGMOID_NOISE),
        ("Gain min", SIG_GAIN_MIN),
        ("Gain max", SIG_GAIN_MAX),
        ("Inflection point (m0)", SIG_X0),
        ("Slope (k)", SIG_SLOPE),
        ("Averaging window (licks)", SIG_AVG_LICKS),
        ("Update every (licks)", SIG_UPDATE_EVERY),

        ("FREE LICKING SESSIONS", None),
        ("No-lick window (s)", f"{_ref_fl.no_lick_wind[0]}–{_ref_fl.no_lick_wind[1]}"),
        ("Reward probability", _ref_fl.reward_prob),
        ("Reward size", _ref_fl.reward_size),
        ("Forced reward — Session 1 (s)", _ref_fl.forced_reward[1]),
        ("Forced reward — Session 2", "none"),

        ("WHISKER DETECTION TASK (WDT1–10)", None),
        ("Number of WDT sessions", NUM_WDT),
        ("No-lick window (s)", f"{_ref_wdt.no_lick_wind[0]}–{_ref_wdt.no_lick_wind[1]}"),
        ("Inter-trial interval (s)", f"{_ref_wdt.iti[0]}–{_ref_wdt.iti[1]}"),
        ("Response window (s)", _ref_wdt.response_wind),
        ("Reward probability", _ref_wdt.reward_prob),
        ("Reward size", _ref_wdt.reward_size),
        ("Trial types (stimulus amplitudes)", DEFAULT_TYPES),
        ("Block size (trials)", _ref_wdt.block_numb),

        ("WDT PSYCHOMETRIC TEST", None),
        ("Trial types (stimulus amplitudes)", WDT_TEST_TYPES),

        ("POPULATION / COHORT SIMULATION", None),
        ("Number of mice", N_MICE),
        ("Noise gain", NOISE_GAIN),
        ("Stimulus learning rate", LEARNING_STIM),
    ]

    save_parameters_txt(param_rows)
