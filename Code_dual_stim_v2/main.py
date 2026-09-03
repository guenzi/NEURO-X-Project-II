# Deroulement : 2 sessions Free Licking + N sessions de detection a deux stimuli
# (whisker=stim1, auditif=stim2), avec un switch de contingence a mi-parcours :
# sessions 1..SWITCH_SESSION-1 -> stim1 recompense (etiquette "WDT"), sessions
# SWITCH_SESSION..NUM_WDT -> stim2 recompense (etiquette "AUD"). Reconstruit sur le
# squelette de Code_mono_stim_v2 (meme Mouse, meme decision D, meme desapprentissage
# general), avec en plus un desapprentissage SPECIFIQUE au stimulus (voir models.py).
#
# DELEARNING=True active a la fois le desapprentissage general (impulsivite) et le
# desapprentissage specifique au stimulus (le vrai test : stim1 doit s'eteindre apres
# le switch, sans affecter stim2).

from models import SimConfig
from functions import (
    initialization,
    run_FL1,
    run_FL2,
    run_all_wdt,
    plot_all_results,
    save_run_parameters,
)

config = SimConfig(
    DELEARNING=True,     # False = comportement d'avant le desapprentissage (E et gains de
                          # stimulus ne peuvent que monter, jamais s'eteindre specifiquement)
    NUM_WDT=10,
    SWITCH_SESSION=6,     # sessions 1-5 : stim1 (whisker) recompense ; 6-10 : stim2 (auditif)
)

session_info, mouse = initialization(config)

log_fl1 = run_FL1(mouse, session_info, config)
log_fl2 = run_FL2(mouse, session_info, config, log_fl1)

wdt_bundle = run_all_wdt(mouse, session_info, config, log_fl2)

plot_all_results(session_info, mouse, config, log_fl1, log_fl2, wdt_bundle)

if config.SAVE_PARAMETERS_TXT:
    save_run_parameters(config, mouse, session_info)
