# Virtual mouse model

Simulation of a mouse learning a detection task (Whisker Detection Task, WDT) through
reinforcement. The goal is to reproduce learning curves, the effect of noise,
delearning/relearning, and inter-individual variability observed in real mice, with a
simple model (Expectation + thresholded decision).

## Structure of a simulation

A simulation runs bin by bin (0.1s by default, see `SessionBaseInfo`) over a sequence of
sessions:

1. **Free Licking 1 (FL1)** — the mouse has no notion of the task yet. It licks at
   random (noise); at a fixed time (500s), a reward is forced even without a lick,
   to teach it that licking can pay off.
2. **Free Licking 2 (FL2)** — no more forced reward, only spontaneous licks rewarded
   according to probability. Consolidates what FL1 started.
3. **WDT1 through WDT10** — the actual training sessions. Each trial presents either a
   stimulus (to be detected by licking within a response window) or a catch (nothing,
   don't lick). This is where the learning curve is built.
4. **WDT_TEST** — a final session with several stimulus amplitudes (instead of a single
   one), to trace a psychometric curve (Hit Rate as a function of amplitude) on an
   already-trained mouse. Absent in whisker/auditory, which has its own probe mechanism
   (the contingency switch, see below).

Expectation, Uncertainty, and the No-Go pathway (see below) persist from one session to
the next: each new session starts with the final state of the previous one, not from
zero.

## The decision to lick

At each bin, the mouse can emit a lick. The rule is the same across all three modes
(mono, dual, whisker/auditory):

$$
E = E_{go} - E_{nogo}
$$

$$
D = \left[ \frac{E}{1 + e^{A(U-U_0)}} + \frac{N_d}{1 + e^{-A(U-U_0)}} \right] \times (V \cdot M - C)
$$

$$
\text{lick if } D \geq \text{threshold}
$$

- **E_go** is the reward expectation (in [0, 1]), the learned signal; **E_nogo** is the
  inhibition pathway (see below) — it is always their difference E that enters the
  decision, never E_go alone.
- **N_d** is Gamma noise, independent of expectation.
- **U** (Uncertainty) drives the trade-off between the two terms: when U is low
  (confident mouse, no recent surprise), E dominates strongly; when U is high (recent
  surprises, extinction underway...), noise takes over and the mouse explores more.
- **U_0** (`uncertainty_offset`) is the neutral point of this trade-off: at U = U_0, the
  two terms weigh 50/50. Below it, E dominates; above it, noise dominates.
- **V·M − C** is a multiplicative gate: V (reward value) and C (cost of licking) are
  fixed by default, M (motivation) decreases with each confirmed reward over the course
  of a session — this is the intra-session satiety effect (see below).

## Go/No-Go architecture

In addition to E_go (which rises on reward, with a very long tau ~2000s, near-permanent,
and drops quickly on each unrewarded lick, tau ~4s), a second pathway **E_nogo**
accumulates an inhibition signal, slower and more persistent. The decision uses
**E_go − E_nogo** instead of E_go alone.

E_nogo behaves as a recurrence (recomputed at each bin from its own value at the
previous bin), not as an injection of a future kernel: at the start of each unrewarded
lick bout (not at every bin of the same bout, only the first), it receives a saturating
kick `gain·(1 − E_nogo)`. On each confirmed reward, a fraction (`nogo_relief`) of the
already-accumulated distrust is wiped out — this is what allows fast reacquisition
after extinction.

**Sensitization**: after a series of consecutive unrewarded bouts *within a trial*
(4 by default, `NOGO_STREAK_THRESHOLD`), both the tau **and** the gain of E_nogo grow
— the tau makes it more persistent, the gain makes it more reactive to the next bout.
The two together (rather than either alone) give the deepest suppression during
prolonged extinction, without changing anything about normal learning: this trigger only
counts bouts that fall within a trial's response window, not spontaneous licking between
trials, so it almost never activates before a real delearning is underway.

In practice, with a default delearning setup (`DELEARNING_FROM_SESSION=6`), this
mechanism does fire, repeatedly within a session (verified by directly tracking the
change in `tau_nogo`/`gain_nogo`, not just the value of `nogo_streak_cnt` — which resets
to 0 within the *same* call that reaches the threshold, so it is invisible if you only
look at its value afterwards). On a typical run: `tau_nogo` stays at 8.0 through WDT1-5
(inert outside delearning, as expected), then climbs to 21 as early as WDT6, 28 at WDT7,
32 at WDT8, ending around ~39 at WDT10; `gain_nogo` saturates at its ceiling
(`NOGO_GAIN_MAX=0.6`) as early as WDT9. Sensitization is therefore a real and
significant contributor to the depth of extinction, not a dormant mechanism.

## Stimulus gain, and why it must also be able to decrease

When a stimulus is presented, it injects an anticipatory boost into E_go, proportional
to a learned **stimulus gain** (mono: `exp_update_stim[1]`; dual: `stim_gain_right/left`;
whisker/auditory: `exp_update_stim1_wa/2_wa`). This gain grows with each confirmed
reward (delta rule on RPE).

Without a mechanism for it to decrease, this gain would stay frozen at its learned value
once high and keep injecting the same boost even in the middle of extinction — which
would limit the observable depth of extinction, independently of what the No-Go pathway
already does. The model handles this differently depending on the mode:

- **Mono/dual** (`Mouse.stim_gain_noreward_active`): the gain decreases, with exactly
  the same delta rule as its rise, but ONLY when a correct lick (right side, within the
  response window, after the 2s post-reward cooldown) is not rewarded because the reward
  probability draw failed. With `reward_prob=1.0` (normal behavior outside of targeted
  delearning), this draw can structurally never fail — the mechanism therefore stays
  entirely inert as long as no delearning is underway, and never triggers on a simple
  mismatch (wrong side) or on the cooldown alone.
- **Whisker/auditory**: always active (not optional), because this paradigm always has
  a "wrong" stimulus present (the one no longer rewarded after the contingency switch)
  — the gain of that stimulus decreases directly as soon as a lick on it is not
  rewarded, with no probability draw: it is certain as soon as the wrong stimulus is
  presented.

## The three modes

### Mono-stimulus (`dual_stim=False`, `WHISKER_AUD_STIM=False`)

A single stimulus, a single Expectation E_go, a single E_nogo pathway. The base model,
the one the other two modes were built on without changing anything in it.

### Dual left/right (`dual_stim=True`)

Two stimuli, each leading to a reward on a different side. Two fully independent
Expectations E_right/E_left (each behaves exactly like the mono Expectation, same
update rules), and two independent E_nogo pathways.

Side selection does not directly compare E_right and E_left: independent Gumbel noise
is added to each, and the chosen side is the one with the larger noisy value
(*Gumbel-max trick*) — the exact statistical equivalent of a softmax/logistic choice
rule over `E_right − E_left` (a standard way to introduce stochastic choice between two
options), but with the noise living separately on each representation rather than being
injected directly at the comparison step — even when one side is very confident, there
is always a residual probability of reading the other.

Two additional effects, specific to dual:

- **`cross_stim_gain`**: a stimulus on one side also (weakly) pushes the Expectation of
  the opposite side, to simulate that a naive mouse does not perfectly discriminate the
  two stimuli at first. This relative weight decreases on its own with training since
  the sides' own gains grow while this term stays fixed.
- **`stim_gain_noreward_active`**: see previous section.

### Whisker/auditory (`WHISKER_AUD_STIM=True`, takes priority over `dual_stim` if both
are active by mistake)

Two stimuli (whisker and auditory) but **a single** shared Expectation (as in mono),
each with its own anticipatory gain. The rewarded stimulus switches partway through
training (`WA_SWITCH_SESSION`, session 6 by default) — before it, whisker is rewarded
and auditory never is; after it, the reverse. This contingency switch is what serves as
the behavioral probe (no separate WDT_TEST here).

The licking decision directly reuses the mono function (a single shared Expectation, so
a single shared E_nogo pathway too — not one per stimulus). There is no separate
competing No-Go pathway per stimulus: it is each stimulus's gain that decreases directly
(see previous section, same principle as `stim_gain_noreward_active`, but triggered by
the contingency switch rather than a refused probability draw).

## Delearning and relearning

Not a separate scenario: any mono or dual run can incorporate delearning by setting
`DELEARNING_FROM_SESSION` (e.g. 6: no more reward from WDT6 onward, until the end of WDT
training). In dual, `DELEARNING_SIDE` (+1 right, -1 left) can target a single side, with
the other continuing normally. `DELEARNING_UNTIL_SESSION` restores reward from a given
session onward (relearning) instead of letting extinction last until the end.

As soon as `DELEARNING_FROM_SESSION` is set, three additional diagnostic plots are
generated (`delearning/` folder): Hit Rate per trial around the transitions, E_go vs
E_nogo over time, and average Hit Rate per session as bars colored by phase.

Whisker/auditory does not have this option: its contingency switch already serves as an
extinction/relearning mechanism intrinsic to the paradigm.

## Population simulation

Two distinct mechanisms, not to be confused:

- **`POPULATION_RANGE=True`** (main mode, mono and dual): replaces the entire normal run
  with `run_population()`, which simulates `POPULATION_N_MICE` independent mice (their
  `learning_stim` and the scale of their noise are drawn uniformly within
  ±`POPULATION_PARAM_SPREAD` around the base values) and only produces comparative
  plots — no individual traces. With `POPULATION_SPREAD_SWEEP=True`, the run repeats for
  each percentage in `POPULATION_SWEEP_VALUES` (0/10/20/30/40/50% by default), each in
  its own `population/p{percentage}/` subfolder.
- **`PLOT_STOCHASTIC_IN_POPULATION`** (mono only, **disabled by default**, an add-on to
  a normal run): if enabled, at the end of a normal mono run it reruns a small
  independent cohort (`N_MICE`/`NOISE_GAIN`/`LEARNING_STIM`/`SESSION_NAME`), simpler than
  the full population mode, just to visualize Hit Rate/False Alarm variability on a given
  session. Disabled by default because its engine
  (`simulate_mouse_and_get_session_perf`) builds its mouse via a bare `Mouse()`, without
  going through `_build_mouse`/`SimConfig` — several parameters (cost, lick thresholds,
  No-Go/sensitization block) are left at their dataclass defaults there, different from
  the values used everywhere else in the project.

## Effect of intra-session Motivation

The raw Hit Rate often plateaus below what E_go has actually learned, because Motivation
(M) decreases with each confirmed reward over the course of a session — the `V·M − C`
gate progressively shrinks, until only an Expectation close to its maximum stays above
the lick threshold. This is not a learning deficit but a satiety effect.

To isolate the "pure" performance from this effect, each trial is recomputed as if M had
stayed at 1.0 for the whole session (`D_norm = D · (V−C)/(V·M−C)`, compared to the same
threshold) — this normalized plot is generated automatically on every run, mono as well
as dual, population or not.

## Files

- **`models.py`** — all data structures: `Mouse` (parameters and learned gains),
  `SimConfig` (all the settings for a run), the session classes (params/state) for FL,
  WDT, and their dual/whisker-auditory equivalents.
- **`functions.py`** — all the simulation logic (one function per decision/update bin)
  and the orchestration functions called by `main.py` (one per mode and session type).
- **`plotting.py`** — all the plots (traces, learning curves, delearning diagnostics,
  population...) and management of the `results/` output folder.
- **`main.py`** — builds a `SimConfig` and calls the right orchestration function based
  on its flags. See the scenario table at the top of the file for common combinations
  (mono/dual/whisker-auditory, with or without delearning, with or without population).

## Results

Each run creates a folder `results/{date}_{time}_{mode}[_delearnFrom{N}[_{side}]][_until{M}]/`
(mono/dual/whiskAud depending on the mode, delearning suffix if active), with a
subfolder per plot category (`traces/`, `block_rates/`, `learning_curve/`,
`session_summary/`, `rpe/`, `psychometric/`, `delearning/`, `population/`) and, if
`SAVE_PARAMETERS_TXT` is active, a `parameters.txt` listing every parameter of the run.

## Environment

The conda environment is defined in `bio482.yml`: Python 3.11, numpy, matplotlib, scipy.

```bash
conda env create -f bio482.yml
conda activate bio482
```
Then select the Python Interpreter in the search bar while in the file. Search bar ->
Cmd + P -> Python: Select Interpreter -> bio482.

## Example run

The simplest case: mono-stimulus, default settings.

```python
from models import SimConfig
from functions import initialization, run_FL1, run_FL2, run_all_wdt, run_wdt_test, plot_all_results

config = SimConfig()

session_info, mouse = initialization(config)
log_fl1 = run_FL1(mouse, session_info, config)
log_fl2 = run_FL2(mouse, session_info, config, log_fl1)
wdt_bundle = run_all_wdt(mouse, session_info, config, log_fl2)
wdt_test = run_wdt_test(mouse, session_info, config, wdt_bundle)

plot_all_results(session_info, mouse, config, log_fl1, log_fl2, wdt_bundle, wdt_test)
```

Results (plots + `parameters.txt`) are written to `results/{date}_{time}_mono/`.
