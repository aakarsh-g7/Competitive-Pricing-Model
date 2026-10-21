import os
import pickle
import numpy as np
import pandas as pd


'''
Part 1 agent: Personalized pricing under capacity constraints using 
Dynamic Programming / Bellman backwards induction.
'''

# Precomputed Bellman opportunity cost table for T_rem in [0..20] and k in [0..20]
FALLBACK_OPP_COST_TABLE = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 59.0045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 78.9256, 39.0834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 95.8453, 49.1333, 32.0349, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 111.1287, 56.3604, 41.0009, 27.5279, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 125.1887, 62.8198, 46.4345, 36.4655, 24.1139, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 138.234, 68.9587, 50.735, 41.4918, 33.2647, 21.3426, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 150.4829, 74.9356, 54.5089, 45.1967, 38.1676, 30.7147, 19.0249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 162.0232, 80.8056, 58.0679, 48.2153, 41.6703, 35.6602, 28.5484, 17.0449, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 172.896, 86.646, 61.4821, 50.9183, 44.4556, 39.0556, 33.6229, 26.6359, 15.3278, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 183.1398, 92.5299, 64.7768, 53.4408, 46.8224, 41.729, 36.9981, 31.8794, 24.9039, 13.8247, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 192.7912, 98.448, 68.0653, 55.8273, 48.9151, 43.9786, 39.598, 35.281, 30.3277, 23.3182, 12.4989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 201.8842, 104.4244, 71.3522, 58.1171, 50.8543, 45.9187, 41.7614, 37.845, 33.7981, 28.916, 21.8593, 11.323, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 210.4513, 110.4435, 74.6521, 60.3474, 52.7017, 47.6423, 43.6447, 39.9544, 36.3582, 32.4751, 27.6029, 20.5095, 10.275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 218.5229, 116.4843, 77.9865, 62.5426, 54.5011, 49.2369, 45.2863, 41.7754, 38.4368, 35.0605, 31.2715, 26.3662, 19.2548, 9.3369, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 226.1276, 122.5442, 81.3524, 64.7443, 56.2401, 50.7409, 46.7607, 43.3896, 40.2077, 37.1217, 33.9038, 30.159, 25.1955, 18.0852, 8.4944, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 233.2925, 128.6118, 84.776, 66.9514, 57.9248, 52.1909, 48.1233, 44.8241, 41.7772, 38.8692, 35.96, 32.8401, 29.1193, 24.084, 16.9907, 7.7362, 0.0, 0.0, 0.0, 0.0],
    [0.0, 240.0429, 134.6691, 88.2786, 69.1632, 59.5624, 53.6121, 49.4055, 46.1102, 43.1909, 40.3951, 37.6947, 34.921, 31.857, 28.1285, 23.0281, 15.9644, 7.0523, 0.0, 0.0, 0.0],
    [0.0, 246.4028, 140.7643, 91.7938, 71.3891, 61.1688, 55.0093, 50.6244, 47.3029, 44.4667, 41.7743, 39.1976, 36.6393, 33.9739, 30.9393, 27.1756, 22.023, 15.0012, 6.4344, 0.0, 0.0],
    [0.0, 252.3949, 146.8747, 95.3283, 73.633, 62.7692, 56.3723, 51.813, 48.4294, 45.6139, 43.032, 40.5398, 38.1372, 35.6899, 33.0897, 30.0758, 26.2567, 21.0632, 14.0965, 5.8753, 0.0]
])


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.project_part = params.get('project_part', 1)
        self.inventory_replenish = params.get('inventory_replenish', 20)
        self.remaining_inventory = params.get('inventory_limit', 20)
        if isinstance(self.remaining_inventory, dict):
            self.remaining_inventory = self.remaining_inventory.get('max', 20)

        # Load trained demand model
        self.filename = os.path.join(os.path.dirname(__file__), 'helper_files_here', 'purchase_model.pkl')
        with open(self.filename, 'rb') as file:
            self.trained_model = pickle.load(file)

        # Load DP opportunity cost table
        opp_cost_path = os.path.join(os.path.dirname(__file__), 'helper_files_here', 'opp_cost_table.npy')
        if os.path.exists(opp_cost_path):
            try:
                self.opp_cost_table = np.load(opp_cost_path)
            except Exception:
                self.opp_cost_table = FALLBACK_OPP_COST_TABLE
        else:
            self.opp_cost_table = FALLBACK_OPP_COST_TABLE

        # Price search grid for candidate personalized prices
        self.price_grid = np.linspace(0, 350, 176)

    def _process_last_sale(
            self,
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        Updates internal state metrics based on feedback from the simulator.
        '''
        self.remaining_inventory = inventories[self.this_agent_number]

    def action(self, obs):
        '''
        Sets the optimal personalized price for the incoming customer under capacity constraints.

        Obs tuple:
        - new_buyer_covariates: length 3 vector of customer covariates
        - last_sale: feedback tuple from the previous round
        - state: current profit vector
        - inventories: remaining inventory for each agent
        - time_until_replenish: steps remaining until inventory is replenished (1..20)
        '''
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        k = int(inventories[self.this_agent_number])
        t_rem = int(time_until_replenish)

        # No items in stock: do not sell
        if k <= 0:
            return 500.0

        # Marginal opportunity cost of inventory from Bellman Dynamic Program
        if k >= t_rem:
            # Capacity is not binding (more items than remaining customers)
            c = 0.0
        else:
            t_idx = min(max(t_rem, 0), 20)
            k_idx = min(max(k, 0), 20)
            c = float(self.opp_cost_table[t_idx, k_idx])

        # Evaluate predicted purchase probability across candidate prices
        features = pd.DataFrame(
            np.tile(new_buyer_covariates, (len(self.price_grid), 1)),
            columns=['Covariate1', 'Covariate2', 'Covariate3']
        )
        features['price_item'] = self.price_grid

        if hasattr(self.trained_model, 'predict_proba'):
            prob = self.trained_model.predict_proba(features)[:, 1]
        else:
            prob = np.clip(self.trained_model.predict(features), 0, 1)

        # Net surplus above opportunity cost: prob * (price - opportunity_cost)
        surplus = prob * (self.price_grid - c)
        best_idx = np.argmax(surplus)

        if surplus[best_idx] <= 0 and self.price_grid[best_idx] <= c:
            return 500.0

        return float(self.price_grid[best_idx])
