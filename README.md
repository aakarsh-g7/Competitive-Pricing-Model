# Competitive Pricing Model

Personalized pricing that connects purchase prediction with revenue decisions under limited inventory.

**Aakarsh Gundu** · Cornell Tech · ORIE 5355 / INFO 5370: Applied Data Science — Decision Making Beyond Prediction · Fall 2026

## Business problem

What price should a seller offer each customer, and when is it better to save inventory for a future buyer?

This project estimates demand from customer features and offered prices, then uses those estimates to choose prices. It covers unconstrained personalized pricing and a single-seller policy that accounts for inventory and time until replenishment. The repository also includes course starter workflows for competition and a fairness penalty.

## Results and key findings

The [training notebook](model_training/purchase_model_baseline.ipynb) records these purchase-prediction results on a random 80/20 split of 50,000 observations (`random_state=42`):

| Held-out metric | Recorded value |
| --- | ---: |
| Mean squared error | 0.03574 |
| Log loss | 0.11964 |
| ROC-AUC | 0.99145 |

The committed pricing exports also contain these **predicted** revenue averages, calculated as the mean of their `expected_revenue` columns:

| Export | Customers | Predicted average revenue per customer |
| --- | ---: | ---: |
| [Validation prices](model_training/static_prices_validation.csv) | 20,000 | 58.4578 |
| [Test submission](agents/static_prices_submission.csv) | 50,000 | 59.8576 |

These results show strong purchase ranking on one held-out split and a complete set of personalized price recommendations. They do **not** establish realized revenue lift. The static-pricing, capacity, competition, and fairness notebooks have no saved evaluation outputs, and there is no recorded baseline revenue comparison. Test valuations are not supplied. Revenue and prices use the dataset's units; no currency is specified.

## Data

The repository contains course-supplied pricing data. It does not document a real business source or the real-world meanings of the three customer covariates.

| File in `data/` | Rows | Purpose |
| --- | ---: | --- |
| [train_prices_decisions_2026.csv](data/train_prices_decisions_2026.csv) | 50,000 | Customer features, offered price, and purchase decision |
| [validation_user_info_2026.csv](data/validation_user_info_2026.csv) | 20,000 | Customer features for local pricing evaluation |
| [test_user_info_2026.csv](data/test_user_info_2026.csv) | 50,000 | Customer features for the final price submission |

Training columns are `user_index`, `Covariate1`, `Covariate2`, `Covariate3`, `price_item`, and `item_bought`. The purchase decision is Boolean, and `Covariate1` is binary. Validation and test feature files contain only the customer ID and three covariates.

The supplied evaluator uses `validation_valuations_2026.csv` internally to calculate realized validation revenue. The simulator reads encrypted customer covariates and valuations from `datafile1_2026.csv` and `datafile2_2026.csv`. These valuation files are used for evaluation, not as demand-model training features.

## Modeling and pricing approach

1. **Estimate purchase probability.** An `XGBRegressor` with a `binary:logistic` objective uses three covariates and the offered price. It has 350 trees, maximum depth 5, and learning rate 0.03. A monotonic constraint on price makes predicted demand non-increasing as price rises, holding customer features fixed. After held-out evaluation, the final model is refit on all 50,000 training observations.

2. **Choose static personalized prices.** For each customer, maximize `price × predicted purchase probability` over 251 candidate prices from 0 to 500 in increments of 2. The implementation evaluates customers in batches of 2,000 and exports one price and expected revenue per customer.

3. **Account for scarce inventory.** A Bellman dynamic program uses 5,000 sampled training customers to precompute a 21 × 21 opportunity-cost table over remaining customers and inventory. The [Part 1 agent](agents/agent_part1.py) chooses the price maximizing `predicted purchase probability × (price − inventory opportunity cost)` over prices from 0 to 350 in increments of 2. The opportunity cost is zero when inventory covers every remaining customer before replenishment. Default inventory resets to a uniformly random integer from 7 to 20 every 20 customers.

4. **Explore competition and fairness.** The Part 2 starter uses the demand model and an inventory heuristic but does not model the opponent. The optional fairness starter searches blends of personalized and common prices for `gamma = 0, 0.1, 0.9`, with score `average revenue − gamma × absolute gap between group median offered prices`. Matching medians alone does not establish broader fairness.

The price grids are implementation choices. The capacity policy is based on estimated demand and a sampled customer distribution; it is not a claim of globally optimal real-world pricing.

## My contribution

I extended the supplied course framework with:

- **Monotonic XGBoost demand estimation:** replaced the starter linear regression, added held-out MSE/log-loss/ROC-AUC evaluation, and verified predictions after saving and reloading the model.
- **Inventory-aware pricing:** implemented Bellman opportunity-cost precomputation and replaced the Part 1 inventory heuristic with a dynamic pricing policy.
- **Static-price optimization:** refined the price grid and added batched evaluation across customer-price combinations.
- **Saved artifacts and exports:** committed the trained model, opportunity-cost table, validation prices, and test submission.

These changes are visible in the [implementation commit](https://github.com/aakarsh-g7/Competitive-Pricing-Model/commit/ef6482f05c915851d224ecb967e8a0d57f15359e). The data, simulator, evaluation helpers, and competition/fairness starter workflows were part of the initial course scaffold.

## Example output

The first three rows of the committed [test submission](agents/static_prices_submission.csv), with predicted revenue rounded here to four decimals:

| user_index | price_item | expected_revenue |
| ---: | ---: | ---: |
| 50000 | 46.0 | 42.9947 |
| 50001 | 368.0 | 338.2232 |
| 50002 | 368.0 | 314.6325 |

`expected_revenue` is a model estimate, not an observed sale. The export contains every test customer exactly once.

Training saves `purchase_model.pkl` and `opp_cost_table.npy` in [agents/helper_files_here/](agents/helper_files_here/). The capacity and competition notebooks generate cumulative revenue and inventory plots, plus average revenue per customer and cumulative buyer utility. Those simulation outputs are not saved in the committed notebooks.

## Setup and reproduction

Use **Python 3.13** and the versions pinned in [requirements.txt](requirements.txt). Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/aakarsh-g7/Competitive-Pricing-Model.git
cd Competitive-Pricing-Model

uv venv --python 3.13
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv ipykernel
uv pip check --python .venv
```

Open the repository folder in VS Code with the Python and Jupyter extensions. Select its `.venv` Python environment as the notebook kernel. Keep `.venv` out of version control.

Run the training notebook first, then choose a pricing workflow:

| Order | Notebook | What it produces |
| --- | --- | --- |
| 1 | [Purchase-model training](model_training/purchase_model_baseline.ipynb) | Held-out metrics, final model, and Bellman opportunity-cost table |
| 2 | [Static personalized pricing](run_project_part1_static_2026.ipynb) | Validation prices, aggregate validation evaluation, and test submission |
| After 1 | [Capacity-constrained pricing](run_project_part1_capacity_2026.ipynb) | Single-seller inventory simulation and plots |
| After 1 | [Head-to-head competition](run_project_part2_2026.ipynb) | Two-agent simulation; change `agentnames` to select opponents |
| Optional, after 1 | [Fairness experiment](run_project_bonus_fairness_2026.ipynb) | Separate validation and test CSVs for the fairness objective |

Training overwrites the saved model and opportunity-cost table. The static and fairness notebooks overwrite their generated CSVs by default; set `overwrite_csvs = False` to protect existing exports. The fairness CSVs are generated by that workflow and are not currently committed.

To score the existing validation export, run this in a notebook cell from the repository root:

```python
from pricing_helpers import evaluate_static_prices

evaluate_static_prices("model_training/static_prices_validation.csv")
```

The evaluator returns customer count, predicted average revenue, realized average revenue, and score without exposing valuation labels. This is separate from the demand model's 80/20 split. Repeated tuning against the same validation customers can overfit pricing decisions.

Capacity and competition notebooks default to 500 customers. For comparisons, create fresh environments and agents, use several trials, and record the seeds and results; the simulator accepts a seed through `params`. The repository does not yet contain a saved multi-trial comparison.
