# import gym
# from gym import spaces
# from gym.envs.registration import EnvSpec
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from cryptography.fernet import Fernet
import pickle

DATA_ENCRYPTION_KEY = bytes('00000000000002026keyyouseethisbutdonotuseit=', 'utf-8')

# Code adapted from here: https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py
# also useful: https://www.ai-articles.net/creating-a-custom-gym-openai-environment-for-algorithmic-trading/


def encrypt_df_columns(df, list_of_columns, secret_key):
    obj = Fernet(secret_key)
    for col in list_of_columns:
        df[col] = df[col].apply(lambda x: obj.encrypt(
            bytes(str(x).encode('utf-8').hex(), 'utf-8')))
    return df


def decrypt_df_columns(df, list_of_columns, secret_key):
    obj = Fernet(secret_key)
    for col in list_of_columns:
        df[col] = df[col].apply(lambda x: float(bytes.fromhex(
            obj.decrypt(bytes(x[2:-1], 'utf-8')).decode().strip())))

    return df


def read_secret_dataframe(secret_file_name_str, secret_key):
    df = pd.read_csv(secret_file_name_str)

    if secret_key is not None:
        df = decrypt_df_columns(df, df.columns.tolist(), secret_key)
    
    df.index = df['user_index'].values
    del df['user_index']
    return df


def openpickle(filename):
    with open(filename, "rb") as readfile:
        loaded = pickle.load(readfile)
    return loaded

def draw_inventory_limit(inventory_limit, rng=random):
    if isinstance(inventory_limit, dict):
        return rng.randint(inventory_limit["min"], inventory_limit["max"])
    else:
        return inventory_limit

class MultiAgentEnv_algopricing(object):  # gym.Env
    def __init__(
            self, 
            params, 
            agent_names, 
            customer_covariates_file=None, 
            customer_valuations_file=None,
            inventory_limit = {"min": 7, "max": 20},
            inventory_replenish = 20
        ):
        self.time = 0
        self.cumulative_buyer_utility = 0
        self.n_agents = params["n_agents"]
        self.project_part = params["project_part"]
        self.agent_names = agent_names
        self.agent_profits = [0 for _ in range(self.n_agents)]
        self.agent_profits_timeseries = [[] for _ in range(self.n_agents)]
        self.inventory_limit = copy.deepcopy(inventory_limit)
        self.inventory_replenish = inventory_replenish
        # Agent randomization and sale/tie outcomes must not change future customers or inventory.
        seed = params.get('seed')
        if seed is None:
            seed = random.getrandbits(64)
        self.customer_rng = random.Random(f'{seed}:customers')
        self.inventory_rng = random.Random(f'{seed}:inventory')
        self.tie_rng = random.Random(f'{seed}:ties')
        initial_limit = draw_inventory_limit(self.inventory_limit, self.inventory_rng)
        self.remaining_inventory = [initial_limit for _ in range(self.n_agents)]
        self.remaining_inventory_timeseries = [[] for _ in range(self.n_agents)]
        self.customers = []
        self.customer_covariates_file = customer_covariates_file
        self.customer_truevaluations_file = customer_valuations_file

        self.customer_covariates = None
        self.customer_truevaluations = None

        self.secretkey = DATA_ENCRYPTION_KEY

        self._init_data_files()

    def _init_data_files(self):
        if self.customer_covariates_file is None:
            return
        else:
            self.customer_covariates = read_secret_dataframe(
                self.customer_covariates_file, self.secretkey)
            self.customer_truevaluations = read_secret_dataframe(
                self.customer_truevaluations_file, self.secretkey)

    def get_current_customer(self):
        assert self.time <= len(self.customers)
        if len(self.customers) == self.time:  # create new customer
            customerindex = self.customer_rng.choice(
                self.customer_covariates.index.values)
            covariates = self.customer_covariates.loc[customerindex].values
            valuations = self.customer_truevaluations.loc[customerindex].values
            # print((covariates, valuations))
            self.customers.append((covariates, valuations))
        else:  # have alredy created the customer, just retreiving it
            covariates, valuations = self.customers[self.time]
        return covariates, valuations

    def get_current_state_customer_to_send_agents(self, sale=None):
        if sale is None:
            # sale format: (agent_won, [all_agent_prices])
            sale = (np.nan, [np.nan for _ in range(self.n_agents)])
        customer_covariates, customer_valuations = self.get_current_customer()
        state = self.agent_profits
        inventory = self.remaining_inventory
        time_until_replenish = self.inventory_replenish - self.time % self.inventory_replenish
        # Observations are snapshots, not writable references to simulator state.
        return copy.deepcopy((customer_covariates, sale, state, inventory, time_until_replenish))

    def step(self, all_agent_prices):
        # process current actions
        _, valuations = self.get_current_customer()
        max_utility = 0
        max_utility_agent = -1
        best_agents = []
        # loop through agents. get lowest price for agent that has capacity left, keep track of which item if any the customer is buying
        # keep track of the inventory of agents
        for agent in range(self.n_agents):
            if self.remaining_inventory[agent] > 0:
                util = valuations - all_agent_prices[agent]
                if util >= 0:
                    if util > max_utility:
                        max_utility = util
                        best_agents = [agent]
                    elif util == max_utility:
                        best_agents.append(agent)

        # Buy at price = valuation; break exact ties uniformly among stocked sellers.
        max_utility_agent = self.tie_rng.choice(best_agents) if best_agents else -1
        if max_utility_agent >= 0:
            self.agent_profits[max_utility_agent] += all_agent_prices[max_utility_agent]
            self.cumulative_buyer_utility += max_utility
            self.remaining_inventory[max_utility_agent] -= 1
            sale = (
                max_utility_agent,
                all_agent_prices
            )
        else:
            sale = (np.nan, all_agent_prices)
        for agent in range(self.n_agents):
            self.agent_profits_timeseries[agent].append(
                self.agent_profits[agent])
            self.remaining_inventory_timeseries[agent].append(
                self.remaining_inventory[agent])

        # increase time step
        self.time += 1
        if self.time % self.inventory_replenish == 0:
            # replenish inventory
            new_inventory = draw_inventory_limit(self.inventory_limit, self.inventory_rng)
            self.remaining_inventory = [new_inventory for _ in range(self.n_agents)]

        # create and send new customer for next time step
        return self.get_current_state_customer_to_send_agents(sale)

    def reset(self):
        self.time = 0
        self.cumulative_buyer_utility = 0
        self.agent_profits = [0 for _ in range(self.n_agents)]
        self.agent_profits_timeseries = [[] for _ in range(self.n_agents)]
        inventory_initial = draw_inventory_limit(self.inventory_limit, self.inventory_rng)
        self.remaining_inventory = [inventory_initial for _ in range(self.n_agents)]
        self.remaining_inventory_timeseries = [[] for _ in range(self.n_agents)]

        self.customers = []
        self._init_data_files()

    def render(self, close_before=False, mode="human", close=False, time_update=20):
        if self.time % time_update == 0:
            if close_before:
                plt.close()
            for agent in range(self.n_agents):
                name = "Agent {}: {}".format(agent, self.agent_names[agent])
                plt.plot(
                    list(range(self.time)),
                    self.agent_profits_timeseries[agent],
                    label=name,
                )
            plt.legend(frameon=False)
            plt.xlabel("Time")
            plt.ylabel("Profit")
            sns.despine()
            return True

        return False
        # print("Cumulative buyer utility: {}".format(self.cumulative_buyer_utility))
