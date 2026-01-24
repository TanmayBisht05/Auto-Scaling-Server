import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyBrain:
    def __init__(self, ga_params=None):
        """
        ga_params: List of 4 floats optimized by the Genetic Algorithm.
                   [load_low_limit, load_high_limit, cpu_safe_limit, cpu_danger_limit]
        """
        # --- 1. SETUP VARIABLES ---
        # Load Ratio: 0.0 (Empty) to 2.0 (Double Capacity)
        self.load_ratio = ctrl.Antecedent(np.arange(0, 2.1, 0.1), 'load_ratio')
        # CPU: 0% to 100%
        self.cpu = ctrl.Antecedent(np.arange(0, 101, 1), 'cpu')
        # Action: -1 (Remove) to +1 (Add)
        self.action = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'action')

        # --- 2. DEFINE SHAPES ---
        # Default values if GA hasn't run yet
        p = ga_params if ga_params is not None else [0.5, 0.8, 40, 80]

        # Unpack genes
        L_LOW, L_HIGH = p[0], p[1]
        C_SAFE, C_DANGER = p[2], p[3]

        # LOAD shapes
        self.load_ratio['low'] = fuzz.trimf(
            self.load_ratio.universe, [0, 0, L_LOW])
        self.load_ratio['medium'] = fuzz.trimf(
            self.load_ratio.universe, [0, L_LOW, L_HIGH])
        self.load_ratio['high'] = fuzz.trimf(
            self.load_ratio.universe, [L_LOW, L_HIGH, 2.0])

        # CPU shapes
        self.cpu['safe'] = fuzz.trimf(self.cpu.universe, [0, 0, C_SAFE])
        self.cpu['stable'] = fuzz.trimf(
            self.cpu.universe, [0, C_SAFE, C_DANGER])
        self.cpu['critical'] = fuzz.trimf(
            self.cpu.universe, [C_SAFE, C_DANGER, 100])

        # OUTPUT shapes (Fixed)
        self.action['scale_down'] = fuzz.trimf(
            self.action.universe, [-1, -1, 0])
        self.action['hold'] = fuzz.trimf(self.action.universe, [-0.5, 0, 0.5])
        self.action['scale_up'] = fuzz.trimf(self.action.universe, [0, 1, 1])

        # --- 3. RULES ---
        rule1 = ctrl.Rule(
            self.load_ratio['high'] | self.cpu['critical'], self.action['scale_up'])
        rule2 = ctrl.Rule(
            self.load_ratio['low'] & self.cpu['safe'], self.action['scale_down'])
        rule3 = ctrl.Rule(
            self.load_ratio['medium'] | self.cpu['stable'], self.action['hold'])

        self.ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)

    def compute(self, current_load_ratio, current_cpu):
        self.simulation.input['load_ratio'] = np.clip(current_load_ratio, 0, 2)
        self.simulation.input['cpu'] = np.clip(current_cpu, 0, 100)

        try:
            self.simulation.compute()
            return self.simulation.output['action']
        except:
            return 0  # Fallback
