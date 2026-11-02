# TabFM-Logit

Code for the NeurIPS 2026 paper **Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees** (Yingshuo Wang, Xian Sun, Yanhang Li, Zhichao Fan, Zexin Zhuang). [arXiv:2606.26432](https://arxiv.org/abs/2606.26432)

Tabular foundation models such as TabPFN and Mitra predict discrete choices accurately, but their predictions can violate the economic logic that choice models need: demand that rises with price, negative or implausible willingness-to-pay, probability on unavailable alternatives. TabFM-Logit embeds a foundation model's predicted choice probabilities **q** inside a multinomial logit:

```
V_k(x) = V_k^struct(x) + g_k(q(x))
```

- **Stage 1** fits the structural coefficients of the multinomial logit by maximum likelihood, with sign constraints on time and cost, and with g held at zero.
- **Stage 2** freezes them and fits the small correction network g on the foundation model's predictions, which are computed once and held fixed.

Because q is precomputed, g contributes nothing to any attribute derivative. Trade-off ratios such as the value of time are therefore exactly those of the multinomial logit, and cost monotonicity and zero probability on unavailable alternatives hold by construction. The foundation model contributes accuracy on observed choices.

This repository contains the model, the behavioral audit, every baseline and ablation, and one script per table or figure of the paper.

## Installation

The CPU steps (data, training, counterfactuals, tables) need Python 3.10 or later and:

```bash
pip install -r requirements.txt
```

The foundation-model steps (2, 3 and 5 below) run in a separate GPU environment (Python 3.12, CUDA). Install the packages in this order. TabPFN is installed without dependency resolution because its version pins conflict with AutoGluon's:

```bash
pip install "autogluon.tabular[mitra]==1.5.0"
pip install --no-deps "tabpfn==2.2.1"
pip install eval-type-backport "tabpfn-common-utils[telemetry-interactive]" pydantic pydantic-settings
pip install "scikit-learn==1.6.1"
```

Mitra and TabPFN download their model weights on first use; see their licenses.

## Data

The three datasets are public. Download them into `data/raw/` under these names:

| File | Dataset | Source |
|---|---|---|
| `swissmetro.dat` | Swissmetro stated-preference survey (Bierlaire et al., 2001) | [Biogeme data page](https://transp-or.epfl.ch/biogeme-3.2.8/data.html); also [here](https://raw.githubusercontent.com/timothyb0912/pylogit/master/examples/data/swissmetro.dat) |
| `lpmc.dat` | London Passenger Mode Choice (Hillel et al., 2018) | [Biogeme data page](https://transp-or.epfl.ch/biogeme-3.2.8/data.html) |
| `IoT_DCE_Wearables-PLOS_ONE.dta` | IoT-Wearables choice experiment (Johnson et al., 2020) | [UCL Research Data Repository](https://rdr.ucl.ac.uk/articles/dataset/Internet_of_Things_Discrete_Choice_Experiment_Wearables_data/11568759) |

`python -m tabfm_logit.data --dataset all` builds every split and warns if a raw file's SHA-256 differs from the file used in the paper (checksums in `tabfm_logit/data.py`). The splits are 70/15/15:

- **swissmetro, lpmc:** row-level splits, stratified by the chosen alternative.
- **iot_wearables:** split by subject.
- **swissmetro_resp, lpmc_hh:** the same rows split by respondent and by household, so that no person appears in more than one split.

## Reproducing the paper

Run from the repository root, in order. Outputs go to `results/`.

| Step | Script | Hardware | What it does |
|---|---|---|---|
| 1 | `scripts/1_prepare_data.sh` | CPU, minutes | builds all splits |
| 2 | `scripts/2_fm_predictions.sh` | GPU | Mitra and TabPFN predictions: in-context, cross-fitted (5 folds; group folds on the grouped splits), and restricted-context runs for ablation A4 |
| 3 | `scripts/3_audit_raw_fm.sh` | GPU | behavioral audit of the raw foundation models |
| 4 | `scripts/4_train.sh` | CPU, parallel | TabFM-Logit (10 training-bootstrap replicates per cell), its ablations, the feature-augmented MNL, monotonic neural network, TasteNet, knowledge distillation, fixed-q counterfactuals |
| 5 | `scripts/5_frontier.sh` | GPU | recomputed and partial q: refits each foundation model once and evaluates perturbed inputs |
| 6 | `scripts/6_paper.sh` | CPU, minutes | prints every table and quoted number, and writes the figure |

The full pipeline took roughly 20 GPU-hours and 30 CPU-hours. TabPFN is not run on LPMC, whose train+validation context is outside TabPFN's supported size range.

Where each result comes from (`paper/`):

| Paper | Script |
|---|---|
| Table 1, Section 5.1–5.2 numbers | `table1_headline.py` |
| Table 2 | `table2_grouped.py` |
| Section 5.2 intervals, Appendix F | `intervals.py` |
| Table 3 | `table3_counterfactual.py` |
| Table 4, Appendix H | `table4_frontier.py` |
| Section 6, Appendix B.1 and I tables | `ablations.py` |
| Figure 1 (ablation A4) | `figure_degradation.py` |
| Section 7 calibration, Appendix D | `calibration_appendix.py` |
| Section 5.2 baselines, Appendices E and G | `baselines_appendix.py` |
| Section 4 split statistics | `data_stats.py` |

The foundation-model predictions in the paper were produced with the GPU environment above, and training ran single-threaded on CPU. With `requirements.txt`, re-running the TabFM-Logit and feature-augmented MNL runs on cross-fitted q gives results identical to the paper's (checked on a sample of runs). Other runs, such as the TasteNet and monotonic-network baselines, can differ slightly across PyTorch versions and hardware. Every bootstrap interval uses 2,000 resamples from a generator seeded with 0.

## Using TabFM-Logit on your own data

TabFM-Logit needs a structural specification and the foundation model's predicted probabilities for the training, validation and test rows. The specification is a module with `utilities(X) -> (N, K)`, `get_betas()` and `get_trade_off_ratio()`; see `tabfm_logit/baselines/mnl.py`. Training-row predictions should be cross-fitted, so no row's prediction comes from a model that saw its label.

```python
from tabfm_logit.adapter import train_two_stage

model = train_two_stage(
    "my_data", X_train, y_train, q_train, X_val, y_val, q_val,
    mnl=(MyMNL, n_alternatives), seed=0, bootstrap=False,
)
probs = model.predict_proba_np(X_test, q_test)
ratios = model.mnl.get_trade_off_ratio()   # identical to the Stage 1 multinomial logit's
```

To audit any predictor (monotonicity, finite-difference trade-off ratios, availability leak), use `tabfm_logit.audit.audit(predict_fn, dataset, X_test)` with a dataset configuration in `tabfm_logit/audit.py`.

## Repository layout

```
tabfm_logit/
  data.py               raw files -> splits
  adapter.py            TabFM-Logit model and two-stage / joint training
  train.py              training CLI for TabFM-Logit and its ablations
  audit.py              behavioral audit
  audit_fm.py           audit of a raw foundation model
  calibration.py        ECE and post-hoc calibrators
  counterfactual.py     fixed-q aggregate-share counterfactuals
  frontier.py           recomputed and partial q
  predictions.py        foundation-model prediction files
  fm/                   Mitra and TabPFN runners and refits
  baselines/            multinomial logit, feature-augmented MNL, monotonic NN, TasteNet
  run_*.py              baseline training CLIs
scripts/                the reproduction pipeline
paper/                  one script per table or figure
```

## Citation

```bibtex
@inproceedings{wang2026tabfmlogit,
  title     = {Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees},
  author    = {Wang, Yingshuo and Sun, Xian and Li, Yanhang and Fan, Zhichao and Zhuang, Zexin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache 2.0 (see `LICENSE`).
