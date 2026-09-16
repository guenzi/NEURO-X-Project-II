# Modèle de souris virtuelle — état actuel

Simulation d'une souris qui apprend une tâche de détection (Whisker Detection Task, WDT) par
renforcement. Le but est de reproduire les courbes d'apprentissage, l'effet du bruit, le
désapprentissage/réapprentissage et la variabilité inter-individus qu'on observe chez de
vraies souris, avec un modèle simple (Expectation + décision seuillée).

## Déroulement d'une simulation

Une simulation tourne bin par bin (0.1s par défaut, voir `SessionBaseInfo`) sur une suite de
sessions :

1. **Free Licking 1 (FL1)** — la souris n'a encore aucune notion de la tâche. Elle lèche au
   hasard (bruit) ; à un temps fixe (500s), une récompense est donnée de force même sans lick,
   pour lui apprendre que le lick peut payer.
2. **Free Licking 2 (FL2)** — plus de récompense forcée, seulement des licks spontanés
   récompensés selon probabilité. Sert à consolider ce que FL1 a amorcé.
3. **WDT1 à WDT10** — les vraies sessions d'entraînement. Chaque essai présente soit un
   stimulus (à détecter en léchant dans une fenêtre de réponse), soit un catch (rien, ne pas
   lécher). C'est sur ces sessions que la courbe d'apprentissage se construit.
4. **WDT_TEST** — une session finale à plusieurs amplitudes de stimulus (au lieu d'une seule),
   pour tracer une courbe psychométrique (Hit Rate en fonction de l'amplitude) sur une souris
   déjà entraînée. Absente en whisker/auditif, qui a son propre mécanisme de sonde (le switch
   de contingence, voir plus bas).

L'Expectation, l'Uncertainty et la voie No-Go (voir plus bas) persistent d'une session à
l'autre : chaque nouvelle session démarre avec l'état final de la précédente, pas de zéro.

## La décision de lécher

À chaque bin, la souris peut émettre un lick. La règle est la même dans les trois modes
(mono, dual, whisker/auditif) :

$$
E = E_{go} - E_{nogo}
$$

$$
D = \left[ \frac{E}{1 + e^{A(U-U_0)}} + \frac{N_d}{1 + e^{-A(U-U_0)}} \right] \times (V \cdot M - C)
$$

$$
\text{lick si } D \geq \text{seuil}
$$

- **E_go** est l'expectation de récompense (dans [0, 1]), le signal appris ; **E_nogo** est la
  voie d'inhibition (voir plus bas) — c'est toujours leur différence E qui entre dans la
  décision, jamais E_go seule.
- **N_d** est un bruit Gamma, indépendant de l'expectation.
- **U** (Uncertainty) pilote le compromis entre les deux termes : quand U est bas (souris
  confiante, pas de surprise récente), E domine largement ; quand U est haut (surprises
  récentes, extinction en cours...), le bruit prend le relais et la souris explore davantage.
- **U_0** (`uncertainty_offset`) est le point neutre de ce compromis : à U = U_0, les deux
  termes pèsent 50/50. En dessous, E domine ; au-dessus, le bruit domine.
- **V·M − C** est un gate multiplicatif : V (valeur de la récompense) et C (coût du lick) sont
  fixés par défaut, M (motivation) décroît à chaque récompense confirmée au cours d'une
  session — c'est l'effet de satiété intra-session (voir plus bas).

## Architecture Go/No-Go

En plus de E_go (qui monte à la récompense, tau très long ~2000s, quasi permanent, et
redescend vite à chaque lick non récompensé, tau ~4s), une seconde voie **E_nogo** accumule
un signal d'inhibition, plus lent et plus persistant. La décision utilise **E_go − E_nogo**
au lieu de E_go seule.

E_nogo se comporte comme une récurrence (recalculée à chaque bin à partir de sa propre valeur
au bin précédent), pas comme une injection de noyau futur : au début de chaque bout de lick
non récompensé (pas à chaque bin d'un même bout, seulement au premier), elle reçoit un kick
saturant `gain·(1 − E_nogo)`. À chaque récompense confirmée, une fraction (`nogo_relief`) de
la méfiance déjà accumulée est effacée — c'est ce qui permet une réacquisition rapide après
une extinction.

**Sensibilisation** : après une série de bouts non récompensés consécutifs *dans un essai*
(4 par défaut, `NOGO_STREAK_THRESHOLD`), le tau **et** le gain de E_nogo grandissent tous les
deux — le tau la rend plus persistante, le gain la rend plus réactive au bout suivant. Les
deux ensemble (plutôt que l'un ou l'autre seul) donnent la suppression la plus profonde
pendant une extinction prolongée, sans rien changer à l'apprentissage normal : ce
déclenchement ne compte que les bouts qui tombent dans la fenêtre de réponse d'un essai, pas
le léchage spontané entre les essais, donc il ne s'active quasiment jamais avant qu'un vrai
désapprentissage soit en cours.

En pratique, avec un désapprentissage par défaut (`DELEARNING_FROM_SESSION=6`), ce mécanisme se
déclenche bel et bien, plusieurs fois par session (vérifié en trackant directement le changement
de `tau_nogo`/`gain_nogo`, pas seulement la valeur de `nogo_streak_cnt` — qui se remet à 0 dans
le *même* appel qui atteint le seuil, donc invisible si on ne regarde que sa valeur après coup).
Sur un run type : `tau_nogo` reste à 8.0 pendant WDT1-5 (mécanisme inerte hors désapprentissage,
comme attendu), puis grimpe à 21 dès WDT6, 28 en WDT7, 32 en WDT8, pour finir à ~39 en WDT10 ;
`gain_nogo` sature à son plafond (`NOGO_GAIN_MAX=0.6`) dès WDT9. La sensibilisation est donc un
contributeur réel et significatif à la profondeur de l'extinction, pas un mécanisme dormant.

## Le gain de stimulus, et pourquoi il doit aussi pouvoir baisser

Quand un stimulus est présenté, il injecte un boost anticipatoire dans E_go, proportionnel à
un **gain de stimulus** appris (mono : `exp_update_stim[1]` ; dual : `stim_gain_right/left` ;
whisker/auditif : `exp_update_stim1_wa/2_wa`). Ce gain grandit à chaque récompense confirmée
(règle delta sur le RPE).

Sans mécanisme de baisse, ce gain reste figé à sa valeur apprise une fois haute et continue
d'injecter le même boost même en pleine extinction — ce qui freine la profondeur de
l'extinction observable, indépendamment de ce que fait déjà la voie No-Go. Le modèle gère
ça différemment selon le mode :

- **Mono/dual** (`Mouse.stim_gain_noreward_active`) : le gain baisse, avec exactement la même
  règle delta que sa montée, mais UNIQUEMENT quand un lick correct (bon côté, dans la fenêtre
  de réponse, après le cooldown de 2s post-récompense) n'est pas récompensé parce que le
  tirage de probabilité de récompense a échoué. Avec `reward_prob=1.0` (comportement normal
  hors désapprentissage cible), ce tirage ne peut structurellement jamais échouer — le
  mécanisme reste donc totalement inerte tant qu'aucun désapprentissage n'est en cours, et ne
  se déclenche jamais sur un simple mismatch (mauvais côté) ou sur le cooldown à lui seul.
- **Whisker/auditif** : toujours actif (pas optionnel), parce que ce paradigme a en
  permanence un stimulus "faux" (celui qui n'est plus récompensé après le switch de
  contingence) — le gain de ce stimulus redescend directement dès qu'un lick dessus n'est pas
  récompensé, sans tirage de probabilité : c'est certain dès que le mauvais stimulus est
  présenté.

## Les trois modes

### Mono-stimulus (`dual_stim=False`, `WHISKER_AUD_STIM=False`)

Un seul stimulus, une seule Expectation E_go, une seule voie E_nogo. Le modèle de base, celui
sur lequel les deux autres modes ont été construits sans rien y changer.

### Dual gauche/droite (`dual_stim=True`)

Deux stimuli, chacun menant à une récompense d'un côté différent. Deux Expectations
complètement indépendantes E_droite/E_gauche (chacune se comporte exactement comme
l'Expectation mono, mêmes règles de mise à jour), et deux voies E_nogo indépendantes.

Le choix du côté ne compare pas directement E_droite et E_gauche : un bruit Gumbel
indépendant est ajouté à chacune, et le côté retenu est celui de la plus grande valeur
bruitée (*Gumbel-max trick*). C'est l'équivalent statistique exact d'une sigmoïde sur
`E_droite − E_gauche`, mais le bruit vit sur les deux représentations plutôt que d'être
injecté au moment de la comparaison (comme dans Lak et al. 2020) — même quand un côté est
très confiant, il reste toujours une probabilité résiduelle de lire l'autre.

Deux effets supplémentaires, spécifiques au dual :

- **`cross_stim_gain`** : un stimulus d'un côté pousse aussi (faiblement) l'Expectation du
  côté opposé, pour simuler qu'une souris naïve ne discrimine pas parfaitement les deux
  stimuli au début. Ce poids relatif diminue tout seul avec l'entraînement puisque les gains
  propres grandissent alors que celui-ci reste fixe.
- **`stim_gain_noreward_active`** : voir section précédente.

### Whisker/auditif (`WHISKER_AUD_STIM=True`, prioritaire sur `dual_stim` si les deux sont actives par erreur)

Deux stimuli (whisker et auditif) mais **une seule** Expectation partagée (comme en mono),
chacun avec son propre gain d'anticipation. Le stimulus récompensé change en cours
d'entraînement (`WA_SWITCH_SESSION`, par défaut à la session 6) — avant, whisker est
récompensé et auditif ne l'est jamais ; après, l'inverse. C'est ce switch de contingence qui
sert de sonde comportementale (pas de WDT_TEST séparé ici).

La décision de lécher réutilise directement la fonction mono (une seule Expectation partagée,
donc une seule voie E_nogo partagée elle aussi — pas une par stimulus). Il n'y a pas de voie
No-Go compétitrice séparée par stimulus : c'est le gain de chaque stimulus qui descend
directement (voir section précédente, même principe que `stim_gain_noreward_active`, mais
déclenché par le switch de contingence plutôt que par un tirage de probabilité refusé).

## Désapprentissage et réapprentissage

Pas un scénario à part : n'importe quel run mono ou dual peut intégrer un désapprentissage en
renseignant `DELEARNING_FROM_SESSION` (ex. 6 : plus aucune récompense à partir de WDT6,
jusqu'à la fin de l'entraînement WDT). En dual, `DELEARNING_SIDE` (+1 droite, -1 gauche) peut
cibler un seul côté, l'autre continuant normalement. `DELEARNING_UNTIL_SESSION` restaure la
récompense à partir d'une session donnée (réapprentissage) au lieu de laisser l'extinction
durer jusqu'à la fin.

Dès que `DELEARNING_FROM_SESSION` est renseigné, trois plots de diagnostic supplémentaires
sont générés (dossier `delearning/`) : Hit Rate par essai autour des transitions, E_go vs
E_nogo dans le temps, et Hit Rate moyen par session en barres colorées par phase.

Le whisker/auditif n'a pas cette option : son switch de contingence sert déjà de mécanisme
d'extinction/réapprentissage intrinsèque au paradigme.

## Simulation de population

Deux mécanismes distincts, à ne pas confondre :

- **`POPULATION_RANGE=True`** (mode principal, mono et dual) : remplace tout le run normal
  par `run_population()`, qui simule `POPULATION_N_MICE` souris indépendantes (leur
  `learning_stim` et l'échelle de leur bruit sont tirées uniformément à
  ±`POPULATION_PARAM_SPREAD` autour des valeurs de base) et ne produit que des plots
  comparatifs — pas de traces individuelles. Avec `POPULATION_SPREAD_SWEEP=True`, le run se
  répète pour chaque pourcentage de `POPULATION_SWEEP_VALUES` (0/10/20/30/40/50% par défaut),
  chacun dans son propre sous-dossier `population/p{pourcentage}/`.
- **`PLOT_STOCHASTIC_IN_POPULATION`** (mono uniquement, **désactivé par défaut**, annexe à un
  run normal) : si activé, relance à la fin d'un run mono normal une petite cohorte
  indépendante (`N_MICE`/`NOISE_GAIN`/`LEARNING_STIM`/`SESSION_NAME`), plus simple que le mode
  population complet, juste pour visualiser la variabilité Hit Rate/False Alarm sur une session
  donnée. Désactivé par défaut car son moteur (`simulate_mouse_and_get_session_perf`) construit
  sa souris via `Mouse()` nu, sans passer par `_build_mouse`/`SimConfig` — plusieurs paramètres
  (coût, seuils de lick, bloc No-Go/sensibilisation) y diffèrent donc de ceux utilisés partout
  ailleurs dans le projet.

## Effet de la Motivation intra-session

Le Hit Rate brut plafonne souvent en dessous de ce que E_go a réellement appris, parce que la
Motivation (M) diminue à chaque récompense confirmée au fil d'une session — le gate `V·M − C`
rétrécit progressivement, jusqu'à ce que seule une Expectation proche du maximum reste
au-dessus du seuil de lick. Ce n'est pas un défaut d'apprentissage mais un effet de satiété.

Pour isoler la performance "pure" de cet effet, chaque essai est recalculé comme si M était
resté à 1.0 toute la session (`D_norm = D · (V−C)/(V·M−C)`, comparé au même seuil) — ce plot
normalisé est généré automatiquement sur tous les runs, mono comme dual, population ou non.

## Fichiers

- **`models.py`** — toutes les structures de données : `Mouse` (paramètres et gains appris),
  `SimConfig` (tous les réglages d'un run), les classes de session (params/state) pour FL, WDT
  et leurs équivalents dual/whisker-auditif.
- **`functions.py`** — toute la logique de simulation (une fonction par bin de décision/mise à
  jour) et les fonctions d'orchestration appelées par `main.py` (une par mode et par type de
  session).
- **`plotting.py`** — tous les plots (traces, courbes d'apprentissage, diagnostics de
  désapprentissage, population...) et la gestion du dossier de sortie `results/`.
- **`main.py`** — construit un `SimConfig` et appelle la bonne fonction d'orchestration selon
  ses flags. Voir le tableau de scénarios en tête du fichier pour les combinaisons courantes
  (mono/dual/whisker-auditif, avec ou sans désapprentissage, avec ou sans population).

## Résultats

Chaque run crée un dossier `results/{date}_{heure}_{mode}[_delearnFrom{N}[_{côté}]][_until{M}]/`
(mono/dual/whiskAud selon le mode, suffixe de désapprentissage si actif), avec un
sous-dossier par catégorie de plot (`traces/`, `block_rates/`, `learning_curve/`,
`session_summary/`, `rpe/`, `psychometric/`, `delearning/`, `population/`) et, si
`SAVE_PARAMETERS_TXT` est actif, un `parameters.txt` listant tous les paramètres du run.

## Environnement

L'environnement conda est défini dans `bio482.yml` : Python 3.11, numpy, matplotlib, scipy. 

```bash
conda env create -f bio482.yml
conda activate bio482
```
Ensuite séléctionner le Python Interpreter dans la barre de recherche quand vous êtes dans le fichier. Barre de recherche -> Cmd + P -> Python: Select Interpreter -> bio482.

## Exemple de run

Le cas le plus simple : mono-stimulus, réglages par défaut.

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

Les résultats (plots + `parameters.txt`) sont écrits dans `results/{date}_{heure}_mono/`.
