"""
Purpose: Fuzzy Inference System logic. Defines antecedents (load ratio, CPU), consequents (scaling action), and membership functions. Encapsulates the rule base for scale-up, scale-down, and hold decisions.
Usage: Imported by optimizer.py and brain_server.py. Do not execute directly.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyBrain:
    # Default thresholds — overridden when GA supplies 6 params
    DEFAULT_THRESH_UP   =  0.45
    DEFAULT_THRESH_DOWN = -0.35   # Less conservative than the old -0.6

    def __init__(self, ga_params=None):
        """
        ga_params : list of 4 OR 6 floats from the Genetic Algorithm.
            4-param form (legacy): [load_low, load_high, cpu_safe, cpu_danger]
            6-param form (new):    [load_low, load_high, cpu_safe, cpu_danger,
                                    thresh_up, thresh_down]
        """
        # ------------------------------------------------------------------ #
        #  0.  UNPACK PARAMETERS                                              #
        # ------------------------------------------------------------------ #
        if ga_params is not None and len(ga_params) >= 6:
            L_LOW, L_HIGH     = float(ga_params[0]), float(ga_params[1])
            C_SAFE, C_DANGER  = float(ga_params[2]), float(ga_params[3])
            self.thresh_up    = float(ga_params[4])
            self.thresh_down  = float(ga_params[5])
        elif ga_params is not None and len(ga_params) >= 4:
            L_LOW, L_HIGH     = float(ga_params[0]), float(ga_params[1])
            C_SAFE, C_DANGER  = float(ga_params[2]), float(ga_params[3])
            self.thresh_up    = self.DEFAULT_THRESH_UP
            self.thresh_down  = self.DEFAULT_THRESH_DOWN
        else:
            L_LOW, L_HIGH     = 0.5, 0.8
            C_SAFE, C_DANGER  = 40.0, 80.0
            self.thresh_up    = self.DEFAULT_THRESH_UP
            self.thresh_down  = self.DEFAULT_THRESH_DOWN

        # Safety: ensure L_LOW < L_HIGH and C_SAFE < C_DANGER
        L_LOW    = np.clip(L_LOW,    0.05, 1.8)
        L_HIGH   = np.clip(L_HIGH,   L_LOW + 0.1, 2.0)
        C_SAFE   = np.clip(C_SAFE,   5.0,  90.0)
        C_DANGER = np.clip(C_DANGER, C_SAFE + 5.0, 100.0)

        # ------------------------------------------------------------------ #
        #  1.  UNIVERSE OF DISCOURSE                                          #
        # ------------------------------------------------------------------ #
        self.load_ratio = ctrl.Antecedent(np.arange(0, 2.05, 0.05), 'load_ratio')
        self.cpu        = ctrl.Antecedent(np.arange(0, 101,  1),    'cpu')
        self.action     = ctrl.Consequent(np.arange(-1, 1.05, 0.05), 'action')

        # ------------------------------------------------------------------ #
        #  2.  MEMBERSHIP FUNCTIONS                                           #
        # ------------------------------------------------------------------ #

        # --- Load ratio --------------------------------------------------- #
        # FIX: trapmf gives a *flat top* from 0 up to L_LOW*0.7, then slopes
        # down to 0 at L_LOW.  This means any load below ~70 % of L_LOW has
        # full "low" membership — the system can now signal scale-down under
        # realistic traffic, not just at RPS=0.
        low_plateau = round(L_LOW * 0.7, 3)
        self.load_ratio['low'] = fuzz.trapmf(
            self.load_ratio.universe, [0, 0, low_plateau, L_LOW])

        self.load_ratio['medium'] = fuzz.trimf(
            self.load_ratio.universe, [low_plateau, L_LOW, L_HIGH])

        self.load_ratio['high'] = fuzz.trapmf(
            self.load_ratio.universe, [L_LOW, L_HIGH, 2.0, 2.0])

        # --- CPU ---------------------------------------------------------- #
        self.cpu['safe'] = fuzz.trapmf(
            self.cpu.universe, [0, 0, C_SAFE * 0.7, C_SAFE])

        self.cpu['stable'] = fuzz.trimf(
            self.cpu.universe, [C_SAFE * 0.7, C_SAFE, C_DANGER])

        self.cpu['critical'] = fuzz.trapmf(
            self.cpu.universe, [C_SAFE, C_DANGER, 100, 100])

        # --- Action output ------------------------------------------------ #
        self.action['scale_down'] = fuzz.trapmf(
            self.action.universe, [-1, -1, -0.6, -0.1])

        self.action['hold'] = fuzz.trimf(
            self.action.universe, [-0.4, 0, 0.4])

        self.action['scale_up'] = fuzz.trapmf(
            self.action.universe, [0.1, 0.6, 1, 1])

        # ------------------------------------------------------------------ #
        #  3.  RULES                                                          #
        # ------------------------------------------------------------------ #
        # Rule 1 — aggressive scale-up on high load OR critical CPU
        rule1 = ctrl.Rule(
            self.load_ratio['high'] | self.cpu['critical'],
            self.action['scale_up'])

        # Rule 2 — strong scale-down when *both* load and CPU are low
        rule2 = ctrl.Rule(
            self.load_ratio['low'] & self.cpu['safe'],
            self.action['scale_down'])

        # Rule 3 (NEW) — softer scale-down on low load alone, even if CPU
        # is stable (but not critical).  This is the key unlock: we no longer
        # need CPU=safe for scale-down to fire.
        rule3 = ctrl.Rule(
            self.load_ratio['low'] & ~self.cpu['critical'],
            self.action['scale_down'])

        # Rule 4 — hold on medium load or stable CPU
        rule4 = ctrl.Rule(
            self.load_ratio['medium'] | self.cpu['stable'],
            self.action['hold'])

        self.ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.simulation  = ctrl.ControlSystemSimulation(self.ctrl_system)

    # ---------------------------------------------------------------------- #
    #  PUBLIC API                                                             #
    # ---------------------------------------------------------------------- #

    def compute(self, current_load_ratio: float, current_cpu: float) -> float:
        """Return action score in [-1, +1].  Returns 0.0 on computation error."""
        self.simulation.input['load_ratio'] = float(np.clip(current_load_ratio, 0, 2))
        self.simulation.input['cpu']        = float(np.clip(current_cpu, 0, 100))
        try:
            self.simulation.compute()
            return float(self.simulation.output['action'])
        except Exception:
            return 0.0

    def get_thresholds(self):
        """Convenience accessor so brain_server doesn't need a separate read."""
        return self.thresh_up, self.thresh_down