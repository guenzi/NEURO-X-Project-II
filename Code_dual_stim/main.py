# main.py — Two-stim detection with mid-block reward switch
# Top-level imports and aliases used throughout the simulation script.

import numpy as np                    # array ops, numerical calculations
import matplotlib.pyplot as plt       # plotting utilities
import random                         # random seeds / RNG control

# Project data models: container classes for mouse, sessions and parameters.
from models import (
    Mouse,                           # mouse state (motivation, noise, stim gains, ...)
    SessionBaseInfo,                 # global session timing / bin settings
    FreeLickingSessionParams,        # params for free-licking sessions
    FreeLickingSessionState,         # runtime state/log for free-licking
    MouseSessionState,               # per-session mouse log (licks, expectation, ...)
    WDTSesssionParams,               # params for WDT (stim detection) sessions
    WDTSesssionState,                # runtime state/log for WDT sessions
)

# Core simulation functions: session stepping, mouse updates, performance metrics.
from functions import (
    function_fl_session,              # advance one bin of free-licking session
    function_mouse_lick,              # sample/record lick events
    function_update_mouse_state,      # update mouse internal state after events
    function_wdt_session,             # advance one bin of WDT (detection) session
    function_performance_wdt,         # compute HR/FA and related metrics for a WDT session
    new_rpe_buffer,                   # helper to maintain running RPE buffer
    online_sigmoid_update,            # online mapping (expectation -> noise gain)
    extract_rpe_per_lick,             # extract RPEs aligned to lick events
)

# Plotting helpers used by the optional visualization switches.
from plotting import (
    plot_traces,                      # time-series visualization (licks, stim, noise)
    plot_wdt_block_rates,             # HR/FA per trial blocks
    plot_session_rates_by_stim,       # HR(stim1/stim2) & FA(catch) across sessions
    plot_single_session_rates,        # HR vs FA + d' per session
    plot_rpe_per_lick,                # RPEs aligned to licks
    plot_noise_gain_sigmoid,          # visualize sigmoid mapping g(m)
    plot_abs_rpe_per_lick,            # absolute RPE per lick visualization
)

# ------------------------------ PLOT SWITCHES ------------------------------
# Toggle visualizations used at the end of the script.
# Set True to enable the corresponding plot, False to skip it.
PLOT_TRACES: bool = False               # time-series traces (licks, stim, noise)
PLOT_BLOCK_HRFA: bool = False           # hit-rate / false-alarm per trial blocks
MAX_TRIALS_BLOCKS: int = 400            # max trials to use when plotting block rates
PLOT_SESSIONS_COMPARISON: bool = False  # compare HR(stim1/stim2) & FA(catch) across sessions
PLOT_SINGLE_SESSION_ALL: bool = False   # HR vs FA + d' for each session individually
PLOT_STIM_GAINS: bool = False           # plot evolution of stimulus gains across sessions
PLOT_RPE_ALL: bool = False              # RPE aligned to licks (raw)
PLOT_ABS_RPE_ALL: bool = False          # absolute RPE per lick
PLOT_PSYCHO_TEST: bool = False          # psychometric test (not used in this 2-stim version)
PLOT_SIGMOID_MAPPING: bool = False      # visualize the sigmoid mapping used for noise gain

# ------------------------------ General settings ------------------------------
NUM_WDT = 10  # total number of WDT sessions (1..5 reward=stim1; 6..10 reward=stim2)

# Optional deterministic RNG seed for reproducible runs.
"""SEED = 80
random.seed(SEED)
np.random.seed(SEED)"""

# ------------------------------ Noise modulation (global) ------------------------------
USE_SIGMOID_NOISE: bool = True    # enable online sigmoid noise modulation
SIG_GAIN_MIN: float   = 0.06      # minimum noise gain
SIG_GAIN_MAX: float   = 0.35      # maximum noise gain
SIG_X0: float         = 0.30      # sigmoid midpoint (motivation)
SIG_SLOPE: float      = 18.0      # sigmoid steepness
SIG_AVG_LICKS: int    = 8         # running RPE buffer length (avg licks)
SIG_UPDATE_EVERY: int = 1         # update noise gain every N bins


# =============================================================================================
#                                   GLOBAL INIT
# =============================================================================================
session_info = SessionBaseInfo()  # base session timing / bin settings
mouse = Mouse()  # mouse state (motivation, noise, stim gains)

# Noise gain traces (for plotting later if you want)
noise_trace_fl1 = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)  # FL1 trace init
noise_trace_fl2 = None  # FL2 trace placeholder

wdt_noise_traces:  list[np.ndarray] = []  # per-WDT session noise traces
wdt_sessions:      list[WDTSesssionState] = []  # logs for each WDT session
wdt_mice:          list[MouseSessionState] = []  # mouse logs per WDT session
wdt_perfs:         list[np.ndarray] = []  # performance arrays per WDT session

# Labels: 1..5 "WDT", 6..10 "AUD"
wdt_labels: list[str] = [f"WDT{i}" for i in range(1, 6)] + [f"AUD{i}" for i in range(6, NUM_WDT + 1)]  # session labels


# =============================================================================================
#                                        FL — SESSION 1
# =============================================================================================

fl_params = FreeLickingSessionParams()
fl_session = FreeLickingSessionState(); fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
mouse_session = MouseSessionState();    mouse_session.initialize(session_info.number_bin, mouse.motivation[0])

buf_fl1, cnt_fl1 = new_rpe_buffer(SIG_AVG_LICKS)
noise_trace_fl1[0, 0] = float(mouse.noise[2])

for i in range(1, session_info.number_bin):
    r, s1, s2, fl_session = function_fl_session(
        fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
    )
    _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
    mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, r)

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
    r, s1, s2, fl_session = function_fl_session(
        fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
    )
    _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
    mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, r)

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
#                         TWO-STIM DETECTION — SESSIONS 1..10
# =============================================================================================

prev_ms = log_mouse_fl2
prev_rw = log_session_fl2.reward

# ================== WDT — logging des gains de stim ==================
stim_gain_history: list[tuple[float, float]] = []
print(f"Initial stim gains: stim1={mouse.stim_gain1:.3f}, stim2={mouse.stim_gain2:.3f}")


for s_idx in range(1, NUM_WDT + 1):
    wdt_params = WDTSesssionParams()
    wdt_params.trial_kinds = [0, 1, 2]             # ~1/3 each on average
    wdt_params.reward_stim = 1 if s_idx <= 5 else 2  # switch after 5 sessions
    # (stim amps default to 1.0, but you can tweak wdt_params.stim1_amp/stim2_amp)

    lick_sum   = float(np.sum(prev_ms.lick))
    reward_sum = float(np.sum(prev_rw))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0

    wdt_session = WDTSesssionState(); wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState(); mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect)

    # Noise gain trace for this session
    noise_trace_wdt = np.full((session_info.number_bin, 1), float(mouse.noise[2]), dtype=float)
    noise_trace_wdt[0, 0] = float(mouse.noise[2])

    buf_wdt, cnt_wdt = new_rpe_buffer(SIG_AVG_LICKS)

    for i in range(1, session_info.number_bin):
        r, s1, s2, wdt_session = function_wdt_session(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, r)

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
        stim_mask  = perf[:, 1] > 0   # 1 or 2 = stim present
        catch_mask = ~stim_mask
        hr = (np.sum((perf[:, 5] == 1) & stim_mask)  / np.sum(stim_mask))  if np.sum(stim_mask)  > 0 else 0.0
        fa = (np.sum((perf[:, 5] == 3) & catch_mask) / np.sum(catch_mask)) if np.sum(catch_mask) > 0 else 0.0
    else:
        hr, fa = 0.0, 0.0

    label = wdt_labels[s_idx - 1]
    print(f"Hit Rate {label} : {hr:.2f}")
    print(f"FA  Rate {label} : {fa:.2f}")
    
    g1 = float(mouse.stim_gain1)
    g2 = float(mouse.stim_gain2)
    print(f"Stim gains after {label}: stim1={g1:.3f}, stim2={g2:.3f}")
    stim_gain_history.append((g1, g2))

    # store
    wdt_perfs.append(perf)
    wdt_sessions.append(wdt_session)
    wdt_mice.append(mouse_session)
    wdt_noise_traces.append(noise_trace_wdt)

    prev_ms = mouse_session
    prev_rw = wdt_session.reward

# =========================== PLOTTING (driven by switches) ============================

if PLOT_TRACES:
    plot_traces(session_info.time_vector, log_mouse_fl1, log_session_fl1.reward,
                stim_array=None, title="Free Licking — Session 1", noise_trace=noise_trace_fl1)
    plot_traces(session_info.time_vector, log_mouse_fl2, log_session_fl2.reward,
                stim_array=None, title="Free Licking — Session 2", noise_trace=noise_trace_fl2)
    for k, lbl in enumerate(wdt_labels):
        # Visualize both stims on one channel: stim1=1, stim2=2
        stim_vis = wdt_sessions[k].stim1 + 2.0 * wdt_sessions[k].stim2
        plot_traces(
            session_info.time_vector,
            wdt_mice[k],
            wdt_sessions[k].reward,
            stim_array=(wdt_sessions[k].stim1, wdt_sessions[k].stim2),  # <-- both channels
            title=f"Stim Detection — {lbl}",
            noise_trace=wdt_noise_traces[k],
        )


if PLOT_BLOCK_HRFA:
    _wdt_params = WDTSesssionParams()
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_wdt_block_rates(perf, _wdt_params, max_trials=MAX_TRIALS_BLOCKS,
                             title=f"{lbl} — HR/FA per blocks")

if PLOT_SESSIONS_COMPARISON:
    plot_session_rates_by_stim(
        wdt_perfs,
        wdt_labels,
        title="Learning — HR(stim1/stim2) & FA(catch)"
    )


if PLOT_SINGLE_SESSION_ALL:
    for lbl, perf in zip(wdt_labels, wdt_perfs):
        plot_single_session_rates(perf, title=f"{lbl} — HR vs FA")

if PLOT_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl1, title="FL1 — RPE per lick")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_rpe_per_lick(tbl_fl2, title="FL2 — RPE per lick")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        stim_vis = wdt_log.stim1 + 2.0 * wdt_log.stim2
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, stim_vis, session_info)
        plot_rpe_per_lick(tbl, title=f"{lbl} — RPE per lick")

if PLOT_ABS_RPE_ALL:
    tbl_fl1 = extract_rpe_per_lick(log_mouse_fl1, log_session_fl1.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl1, title="FL1 — |RPE| per lick")
    tbl_fl2 = extract_rpe_per_lick(log_mouse_fl2, log_session_fl2.reward, None, session_info)
    plot_abs_rpe_per_lick(tbl_fl2, title="FL2 — |RPE| per lick")
    for lbl, ms_log, wdt_log in zip(wdt_labels, wdt_mice, wdt_sessions):
        stim_vis = wdt_log.stim1 + 2.0 * wdt_log.stim2
        tbl = extract_rpe_per_lick(ms_log, wdt_log.reward, stim_vis, session_info)
        plot_abs_rpe_per_lick(tbl, title=f"{lbl} — |RPE| per lick")

if PLOT_SIGMOID_MAPPING:
    plot_noise_gain_sigmoid(
        m0=SIG_X0,
        k=SIG_SLOPE,
        gmin=SIG_GAIN_MIN,
        gmax=SIG_GAIN_MAX,
        title="Mapping g(m) for noise gain"
    )


if PLOT_STIM_GAINS and stim_gain_history:
    xs = np.arange(1, len(stim_gain_history) + 1)
    g1_arr = np.array([g[0] for g in stim_gain_history], dtype=float)
    g2_arr = np.array([g[1] for g in stim_gain_history], dtype=float)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, g1_arr, marker="o", label="Stim1 gain (WDT)")
    plt.plot(xs, g2_arr, marker="o", label="Stim2 gain (AUD)")
    plt.xticks(xs, wdt_labels)
    plt.xlabel("Sessions")
    plt.ylabel("Gain")
    plt.title("Evolution des gains de stimuli")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
