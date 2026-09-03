"""
Demo du desapprentissage : trois phases (Acquisition -> Extinction -> Reapprentissage)
enchainees sur UNE seule ligne de temps continue, pour montrer sans ambiguite que
E_nogo fait ce qu'il doit faire : la souris ralentit quand la recompense s'arrete, et
reprend quand elle revient.

Chaque phase dure PHASE_DURATION_MIN minutes (15 par defaut) plutot que les 60 minutes
par defaut de SessionBaseInfo : sur une session complete de 60 min, la motivation finit
par s'epuiser (satiete, cf. Mouse.motivation) et la souris arrete de lecher meme si E
reste au maximum -- un phenomene reel et voulu du modele, mais qui n'a RIEN a voir avec
le desapprentissage et polluerait la phase "Acquisition" si on la laissait durer 60 min.

Produit 3 figures distinctes dans results_demo/ :
  1. demo_1_taux_de_leche.png    - taux de leche (moyenne glissante) sur la ligne de temps
                                    complete, phases separees par des pointilles.
  2. demo_2_expectation_vs_nogo.png - E_go, E_nogo et leur difference nette, memes 3 phases.
  3. demo_3_barres_resume.png    - taux de leche moyen par phase (un chiffre par barre).
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from models import SimConfig, SessionBaseInfo, FreeLickingSessionParams, FreeLickingSessionState, MouseSessionState
from functions import initialization, function_fl_session, function_mouse_lick, function_update_mouse_state

SEED = 3
PHASE_DURATION_MIN = 15  # 15 min par phase (900s) au lieu des 60 min par defaut -> evite la satiete
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_demo")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# Architecture Go/No-Go (E_nogo) : toujours active, plus besoin de l'activer explicitement.
config = SimConfig(dual_stim=False, SAVE_PARAMETERS_TXT=False)
_, mouse = initialization(config)  # cree aussi un results/... vide, sans consequence

si = SessionBaseInfo(duration=PHASE_DURATION_MIN)


def run_phase(params, init_expect, init_uncert, init_nogo):
    fls = FreeLickingSessionState(); fls.initialize(si.number_bin, params.no_lick_wind)
    ms = MouseSessionState(); ms.initialize(si.number_bin, mouse.motivation[0], init_expect, init_uncert, init_nogo)
    for i in range(1, si.number_bin):
        r, s, fls = function_fl_session(params, fls, si, int(ms.lick[i - 1, 0]), i)
        _, ms = function_mouse_lick(mouse, ms, i)
        _, ms = function_update_mouse_state(mouse, ms, si, i, s, r)
    return fls, ms


# --- Phase 1 : Acquisition (recompense forcee a t=100s pour demarrer, puis normale) ---
p_acq = FreeLickingSessionParams(); p_acq.forced_reward = (1, 100.0); p_acq.reward_prob = 1.0
fls_acq, ms_acq = run_phase(p_acq, 0.0, 0.0, 0.0)

# --- Phase 2 : Extinction (recompense totalement coupee) ---
p_ext = FreeLickingSessionParams(); p_ext.forced_reward = (0, 0); p_ext.reward_prob = 0.0
fls_ext, ms_ext = run_phase(
    p_ext, float(ms_acq.expectation[-1, 0]), float(ms_acq.uncertainty[-1, 0]), float(ms_acq.expectation_nogo[-1, 0])
)

# --- Phase 3 : Reapprentissage (recompense remise) ---
p_reacq = FreeLickingSessionParams(); p_reacq.forced_reward = (0, 0); p_reacq.reward_prob = 1.0
fls_reacq, ms_reacq = run_phase(
    p_reacq, float(ms_ext.expectation[-1, 0]), float(ms_ext.uncertainty[-1, 0]), float(ms_ext.expectation_nogo[-1, 0])
)

# --- Concatenation sur une ligne de temps continue ---
phases = [("Acquisition", ms_acq, fls_acq), ("Extinction", ms_ext, fls_ext), ("Reapprentissage", ms_reacq, fls_reacq)]
T = si.number_bin
t_full = np.arange(3 * T) * si.resolution
lick_full = np.concatenate([p[1].lick[:, 0] for p in phases])
E_full = np.concatenate([p[1].expectation[:, 0] for p in phases])
Enogo_full = np.concatenate([p[1].expectation_nogo[:, 0] for p in phases])
boundaries = [T * si.resolution, 2 * T * si.resolution]
phase_centers = [T * si.resolution * 0.5, T * si.resolution * 1.5, T * si.resolution * 2.5]
phase_names = ["Acquisition", "Extinction", "Réapprentissage"]

# Taux de leche : moyenne glissante sur une fenetre de 30s (300 bins)
win = 300
lick_rate = np.convolve(lick_full, np.ones(win) / win, mode="same")

# ============================== PLOT 1 : taux de leche dans le temps ==============================
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_full, lick_rate, color="tab:blue")
for b in boundaries:
    ax.axvline(b, linestyle="--", color="gray", alpha=0.7)
ymax = ax.get_ylim()[1]
for c, name in zip(phase_centers, phase_names):
    ax.text(c, ymax * 0.93, name, ha="center", fontsize=11)
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Taux de léchage (moyenne glissante 30s)")
ax.set_title("Taux de léchage : Acquisition → Extinction → Réapprentissage")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "demo_1_taux_de_leche.png"), dpi=150)
plt.close(fig)

# ============================== PLOT 2 : E_go vs E_nogo dans le temps ==============================
# E et E_nogo oscillent a chaque lick (bruit haute frequence) ; on lisse avec la meme fenetre
# que le taux de leche (30s) pour lire la tendance plutot que le dents-de-scie brut.
def smooth(x, w=win):
    return np.convolve(x, np.ones(w) / w, mode="same")

E_smooth = smooth(E_full)
Enogo_smooth = smooth(Enogo_full)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_full, E_smooth, color="tab:blue", label="Expectation (E_go)")
ax.plot(t_full, Enogo_smooth, color="tab:red", label="Expectation No-Go")
ax.plot(t_full, E_smooth - Enogo_smooth, color="black", linewidth=1.5, linestyle="--", label="Net (E_go − E_nogo)")
for b in boundaries:
    ax.axvline(b, linestyle="--", color="gray", alpha=0.7)
for c, name in zip(phase_centers, phase_names):
    ax.text(c, 1.0, name, ha="center", fontsize=11)
ax.set_ylim(-0.15, 1.08)
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Valeur")
ax.set_title("Expectation vs Expectation No-Go : Acquisition → Extinction → Réapprentissage")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "demo_2_expectation_vs_nogo.png"), dpi=150)
plt.close(fig)

# ============================== PLOT 3 : barres résumé par phase ==============================
def mean_rate(ms_x):
    return float(np.mean(ms_x.lick[:, 0]))

rates = [mean_rate(ms_acq), mean_rate(ms_ext), mean_rate(ms_reacq)]
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(phase_names, rates, color=["tab:green", "tab:red", "tab:green"])
for bar, r in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, r + max(rates) * 0.02, f"{r:.3f}", ha="center")
ax.set_ylabel("Taux de léchage moyen (fraction de bins avec un lick)")
ax.set_title("Taux de léchage moyen par phase")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "demo_3_barres_resume.png"), dpi=150)
plt.close(fig)

print("Taux de leche moyen par phase:", dict(zip(phase_names, rates)))
print(f"3 figures sauvegardees dans {OUT_DIR}/")
