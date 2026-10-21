"""Part 2 dummy opponent: adjusts its price after each sale/no-sale."""
class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.alpha = 1.0
        self.project_part = params['project_part'] #useful to be able to use same competition code for each project part

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        did_customer_buy_from_me = last_sale[0] == self.this_agent_number
        if did_customer_buy_from_me:  # can increase prices
            self.alpha *= 1.1
        else:  # should decrease prices
            self.alpha *= 0.9

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer
    # Previous sale, offered prices, profits, inventory, and replenishment time.
    # Returns one scalar price for the single item.

    def action(self, obs):
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        if inventories[self.this_agent_number] == 0:
            return 0
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        return 40*self.alpha
        
