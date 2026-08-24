# functions.py
import numpy as np
import random
from typing import Any, Optional, Tuple

from models import (
    Mouse,
    SessionBaseInfo,
    FreeLickingSessionParams,
    MouseSessionState,
    WDTSesssionParams,
    WDTSesssionState,
    FreeLickingSessionState,
)


def sample_uniform_range(min_val: float, max_val: float) -> float:
    return min_val + (max_val - min_val) * random.random()



def function_mouse_lick(mouse: Mouse, mouse_session: MouseSessionState, i: int) -> tuple[int, MouseSessionState]:
    """
    Decide whether the mouse licks at bin i.
    Uses expectation at i-1, gamma noise (with gain), and current motivation.
    """
    lick = 0

    motivation_t = mouse_session.motivation[i - 1, 0]
    expectation_tm1 = mouse_session.expectation[i - 1, 0]

    # Gamma noise (shape=a, scale=b) scaled by current gain
    gamma_noise = np.random.gamma(mouse.noise[0], mouse.noise[1]) * mouse.noise[2]

    p_lick_t = (expectation_tm1 + gamma_noise) * motivation_t
    if p_lick_t >= mouse.lick_thrs:
        lick = 1

    mouse_session.p_lick[i, 0] = float(p_lick_t)
    mouse_session.lick[i, 0] = lick
    return lick, mouse_session



def function_update_mouse_state(
    mouse: Mouse,
    mouse_session: MouseSessionState,
    session_info: SessionBaseInfo,
    i: int,
    stim1_t: float,
    stim2_t: float,
    reward_t: float
) -> tuple[Mouse, MouseSessionState]:
    """
    Single expectation signal, deux canaux de stimulus (stim1, stim2) avec
    gains indépendants. Les gains sont mis à jour UNIQUEMENT à la récompense
    en utilisant les éligibilités spécifiques de canal (eligibility1/2) au temps i.
    """
    t = i * session_info.resolution
    time_vector = session_info.time_vector

    # --- Motivation
    if reward_t > 0:
        mouse_session.motivation[i, 0] = (
            mouse_session.motivation[i - 1, 0] - mouse.motivation[1] * float(reward_t)
        )
    else:
        mouse_session.motivation[i, 0] = mouse_session.motivation[i - 1, 0]

    # --- RPE (avec expectation au temps i-1)
    rpe_t = float(reward_t) - float(mouse_session.expectation[i - 1, 0])
    mouse_session.rpe[i, 0] = float(rpe_t)

    # --- Contributions des stimuli sur l'expectation + éligibilités
    if (stim1_t > 0.0) or (stim2_t > 0.0):
        tau1, g1 = mouse.exp_update_stim1
        tau2, g2 = mouse.exp_update_stim2

        # Injections exponentielles (causales) dans expectation
        upd1 = (stim1_t * g1 * np.exp(-np.maximum(time_vector - t, 0.0) / tau1)) if stim1_t > 0.0 else 0.0
        upd2 = (stim2_t * g2 * np.exp(-np.maximum(time_vector - t, 0.0) / tau2)) if stim2_t > 0.0 else 0.0
        if np.ndim(upd1) > 0 or np.ndim(upd2) > 0:
            update_exp = (0.0 if np.ndim(upd1) == 0 else upd1) + (0.0 if np.ndim(upd2) == 0 else upd2)
            mouse_session.expectation[i:, 0] += update_exp[i:]  # type: ignore[index]
            mouse_session.expectation = np.clip(mouse_session.expectation, 0.0, 1.0)

        # Éligibilité globale (optionnelle, conservée)
        upd_elig = (stim1_t + stim2_t) * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
        if np.ndim(upd_elig) > 0:
            mouse_session.eligibility[i:, 0] += upd_elig[i:]  # type: ignore[index]
            mouse_session.eligibility = np.clip(mouse_session.eligibility, 0.0, 1.0)

        # Éligibilités par canal (clé pour l'apprentissage correct)
        if stim1_t > 0.0:
            upd_elig1 = stim1_t * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
            mouse_session.eligibility1[i:, 0] += upd_elig1[i:]  # type: ignore[index]
        if stim2_t > 0.0:
            upd_elig2 = stim2_t * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
            mouse_session.eligibility2[i:, 0] += upd_elig2[i:]  # type: ignore[index]

        mouse_session.eligibility1 = np.clip(mouse_session.eligibility1, 0.0, 1.0)
        mouse_session.eligibility2 = np.clip(mouse_session.eligibility2, 0.0, 1.0)

    # --- Expectation : mises à jour reward / no-reward lick
    if reward_t > 0.0:
        upd = rpe_t * mouse.exp_update_reward[1] * np.exp(
            -np.maximum(time_vector - t, 0.0) / mouse.exp_update_reward[0]
        )
        mouse_session.expectation[i:, 0] += upd[i:]  # type: ignore[index]
        mouse_session.expectation = np.clip(mouse_session.expectation, 0.0, 1.0)

    if (reward_t == 0.0) and (mouse_session.lick[i - 1, 0] == 1):
        upd = rpe_t * mouse.exp_update_no_reward[1] * np.exp(
            -np.maximum(time_vector - t, 0.0) / mouse.exp_update_no_reward[0]
        )
        mouse_session.expectation[i:, 0] += upd[i:]  # type: ignore[index]
        mouse_session.expectation = np.clip(mouse_session.expectation, 0.0, 1.0)
        mouse_session.non_rew_lick_cnt += 1

    # --- Apprentissage : ralentir la dérive no-reward si trop de licks
    if mouse_session.non_rew_lick_cnt > mouse.learning_nonrew_lick[0]:
        old_tau, gain = mouse.exp_update_no_reward
        mouse.exp_update_no_reward = (old_tau + mouse.learning_nonrew_lick[1], gain)
        mouse_session.non_rew_lick_cnt = 0

    # --- Apprentissage des gains par canal (à la reward, via eligibilités canal)
    if reward_t > 0.0:
        elig1 = float(mouse_session.eligibility1[i, 0])
        elig2 = float(mouse_session.eligibility2[i, 0])

        tau1, g1 = mouse.exp_update_stim1
        tau2, g2 = mouse.exp_update_stim2

        g1_new = g1 + rpe_t * mouse.learning_stim * elig1
        g2_new = g2 + rpe_t * mouse.learning_stim * elig2

        # borne inférieure à 0 pour éviter des gains négatifs
        mouse.exp_update_stim1 = (tau1, max(0.0, float(g1_new)))
        mouse.exp_update_stim2 = (tau2, max(0.0, float(g2_new)))

    return mouse, mouse_session



def function_fl_session(
    session_param: FreeLickingSessionParams,
    fl_session: FreeLickingSessionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int
) -> Tuple[float, float, float, FreeLickingSessionState]:
    """
    No stimuli in FL, but we return (reward, s1=0, s2=0, state) for a uniform call pattern.
    """
    t = i * session_info.resolution
    reward = 0.0
    s1 = 0.0
    s2 = 0.0

    # Forced reward
    if session_param.forced_reward[0] == 1 and np.isclose(t, session_param.forced_reward[1], atol=1e-6):
        reward = session_param.reward_size
        fl_session.last_reward_time = t

    # Lick-triggered reward
    if lick == 1:
        if (t - fl_session.last_lick_time) > fl_session.no_lick_wind:
            fl_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
            if random.random() <= session_param.reward_prob:
                reward = session_param.reward_size
                fl_session.last_reward_time = t
        fl_session.last_lick_time = t

    fl_session.reward[i, 0] = float(reward)
    # keep stim1 trace at 0.0 for compatibility
    fl_session.stim1[i, 0] = 0.0

    return float(reward), float(s1), float(s2), fl_session



def function_wdt_session(
    session_param: WDTSesssionParams,
    wdt_session: WDTSesssionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int
) -> Tuple[float, float, float, WDTSesssionState]:
    """
    Trials are started when no-lick and ITI constraints are met.
    Each trial kind ∈ {0 (catch), 1 (whisker), 2 (auditory)} sampled ~1/3 each.
    Reward is delivered only if trial kind == session_param.reward_stim AND the lick
    occurs within the response window.
    """
    t = i * session_info.resolution
    reward = 0.0
    s1 = 0.0
    s2 = 0.0

    # Start a new trial?
    if (
        (t - wdt_session.last_lick_time) > wdt_session.no_lick_wind
        and (t - wdt_session.last_trial_time) > wdt_session.iti
    ):
        wdt_session.trial_times.append(t)
        wdt_session.last_trial_time = t
        wdt_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
        wdt_session.iti = sample_uniform_range(*session_param.iti)

        kind = random.choice(session_param.trial_kinds)  # 0/1/2
        wdt_session.last_trial_kind = int(kind)

        if kind == 1:
            s1 = session_param.stim1_amp
            wdt_session.last_stim_time = t
        elif kind == 2:
            s2 = session_param.stim2_amp
            wdt_session.last_stim_time = t

        # Open response window after any stim (1 or 2)
        if kind in (1, 2):
            window_len = int(round(session_param.response_wind / session_info.resolution))
            end_idx = min(i + window_len, session_info.number_bin)
            wdt_session.reward_window[i + 1:end_idx, 0] = 1  # exclude current bin

    # Lick detection / reward
    if lick == 1:
        wdt_session.last_lick_time = t
        if wdt_session.reward_window[i, 0] > 0.5 and (t - wdt_session.last_reward_time) > 2.0:
            if wdt_session.last_trial_kind == session_param.reward_stim:
                if random.random() <= session_param.reward_prob:
                    reward = session_param.reward_size
                    wdt_session.last_reward_time = t

    # Log
    wdt_session.reward[i, 0] = float(reward)
    if s1 > 0:
        wdt_session.stim1[i, 0] = float(s1)
    if s2 > 0:
        wdt_session.stim2[i, 0] = float(s2)

    return float(reward), float(s1), float(s2), wdt_session



def function_performance_wdt(
    wdt_params: WDTSesssionParams,
    session_info: SessionBaseInfo,
    wdt_session: WDTSesssionState,
    mouse_session: MouseSessionState
) -> np.ndarray:
    sr = int(round(1 / session_info.resolution))
    trials = [t for t in wdt_session.trial_times if t < (session_info.duration * 60 - wdt_params.response_wind)]

    performance = []
    for t in trials:
        pt1 = int(round(t * sr))
        pt2 = min(pt1 + int(round(wdt_params.response_wind * sr)), session_info.number_bin)

        s1_seg = wdt_session.stim1[pt1:pt2, 0]
        s2_seg = wdt_session.stim2[pt1:pt2, 0]
        lick_seg = mouse_session.lick[pt1:pt2, 0]
        reward_seg = wdt_session.reward[pt1:pt2, 0]

        s1 = float(np.max(s1_seg))
        s2 = float(np.max(s2_seg))
        stim_code = 1.0 if s1 > 0.0 else (2.0 if s2 > 0.0 else 0.0)

        lick_detected = bool(np.any(lick_seg))
        reward_detected = bool(np.any(reward_seg))
        latency = float(np.argmax(lick_seg) / sr) if lick_detected else float('nan')

        if stim_code > 0 and not lick_detected:
            outcome = 0  # Miss
        elif stim_code > 0 and lick_detected:
            outcome = 1  # Hit
        elif stim_code == 0 and not lick_detected:
            outcome = 2  # Correct Rejection
        else:
            outcome = 3  # False Alarm

        performance.append([
            float(t),
            float(stim_code),
            int(lick_detected),
            float(latency),
            int(reward_detected),
            int(outcome),
        ])

    return np.array(performance, dtype=float)



def extract_rpe_per_lick(mouse_session: MouseSessionState,
                         reward_trace: np.ndarray,
                         stim_trace: Optional[np.ndarray],
                         session_info: SessionBaseInfo,
                         ignore_last_without_next: bool = True
                        ) -> np.ndarray:
    """
    Returns (N x 6):
      [lick_idx, t_sec, rpe_at_lick, reward_flag, stim_amp, bin_index]
    where rpe_at_lick = reward[i+1] - expectation[i]
    """
    lick = np.asarray(mouse_session.lick).ravel()
    expc = np.asarray(mouse_session.expectation).ravel()
    rew  = np.asarray(reward_trace).ravel()
    stim = np.asarray(stim_trace).ravel() if stim_trace is not None else None

    idxs = np.flatnonzero(lick == 1)
    if ignore_last_without_next:
        idxs = idxs[idxs + 1 < len(rew)]

    if idxs.size == 0:
        return np.zeros((0, 6), dtype=float)

    rpe_vals = rew[idxs + 1] - expc[idxs]
    reward_flag = (rew[idxs + 1] > 0).astype(float)
    stim_amp = (stim[idxs] if stim is not None else np.zeros_like(idxs, dtype=float)).astype(float)
    t_sec = idxs.astype(float) * float(session_info.resolution)
    lick_idx = np.arange(1, len(idxs) + 1, dtype=float)

    out = np.column_stack([
        lick_idx,
        t_sec,
        rpe_vals.astype(float),
        reward_flag,
        stim_amp,
        idxs.astype(float)
    ])
    return out


import numpy as _np
from collections import deque as _deque

def sigmoid_gain(m: float, m0: float, k: float, gmin: float, gmax: float) -> float:
    """g(m) = gmin + (gmax-gmin)/(1 + exp(-k*(m-m0)))."""
    return float(gmin + (gmax - gmin) * (1.0 / (1.0 + _np.exp(-k * (m - m0)))))

def new_rpe_buffer(maxlen: int):
    """Create a deque buffer for |RPE| at recent licks + a lick counter."""
    return _deque(maxlen=int(maxlen)), 0

def online_sigmoid_update(
    mouse: Mouse,
    mouse_session: MouseSessionState,
    i: int,
    buf,
    lick_counter: int,
    enabled: bool,
    m0: float,
    k: float,
    gmin: float,
    gmax: float,
    update_every: int,
) -> int:
    """
    Push |RPE[i]| if there was a lick at i-1, and every 'update_every' licks,
    update noise gain via sigmoid(g(mean(|RPE|))). Only changes mouse.noise[2].
    """
    if not enabled:
        return lick_counter

    if mouse_session.lick[i - 1, 0] == 1:
        buf.append(abs(float(mouse_session.rpe[i, 0])))
        lick_counter += 1

        if (lick_counter % int(update_every)) == 0 and len(buf) > 0:
            m = float(_np.mean(buf))
            new_gain = sigmoid_gain(m, m0=m0, k=k, gmin=gmin, gmax=gmax)
            mouse.noise = (mouse.noise[0], mouse.noise[1], float(new_gain))

    return lick_counter
