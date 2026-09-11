# Monotonicity-constrained survival models for credit risk

**What does a structurally valid PD term structure cost?**

Eight models of default risk on 2.25M Lending Club loans, evaluated on the same
674,272 held-out borrowers, ranked on two axes at once: how well they
discriminate, and how often they emit a term structure that cannot happen.

The short answer, in one line: **a structurally monotone neural hazard model
eliminates impossible term structures entirely and costs 0.0070 to 0.0243 AUC
against per-horizon XGBoost, while costing nothing at all against an
unconstrained network.**

---

## 1. The problem

Basel III and IFRS 9 require a PD term structure to be monotone. The probability
that a loan defaults within 36 months cannot be lower than the probability it
defaults within 24 months, because defaulting by month 24 is a subset of
defaulting by month 36. This is not a modelling preference. It is arithmetic, and
a model that violates it is reporting something that cannot occur.

The standard industry approach is to fit an independent classifier per horizon:
one model for 12 months, another for 24, another for 36. Nothing ties them
together, so nothing stops them crossing.

On 674,272 held-out loans, they cross:

| Approach | Borrowers with an impossible term structure | Largest reversal |
|---|---|---|
| XGBoost, one classifier per horizon | 1.687% (11,378) | 0.151 |
| XGBoost, matched 300k training size | 2.219% (14,964) | 0.155 |
| Logistic, one classifier per horizon | 0.046% (311) | 0.022 |
| Unconstrained neural hazard model | **97.515%** (657,518) | **0.402** |

A largest reversal of 0.151 means some borrower is reported as 15 percentage
points *less* likely to default by 36 months than by 24. The unconstrained
network is not a strawman that was badly fitted; it is the natural thing to build
if you ask a neural network for a cumulative hazard and do not constrain it, and
it fails for essentially every borrower it sees.

This project measures what it costs to make that impossible instead of merely
unlikely.

---

## 2. The eight arms

All arms use the same 70/30 split at seed 42, the same 14-column design matrix
from eight source features, and the same evaluation harness. All are scored on
the identical 674,272-row test set.

From [`results/master_comparison.csv`](results/master_comparison.csv):

| Arm | AUC 12m | AUC 24m | AUC 36m | C-Harrell | C-Uno | IBS 1-60m | PD inversions | n_train |
|---|---|---|---|---|---|---|---|---|
| Logistic (independent horizons) | 0.7012 | 0.6911 | 0.6876 | 0.6804 | 0.6419 | 0.1501 | 0.046% | 1,573,299 |
| Logistic (300k) | 0.7010 | 0.6909 | 0.6875 | 0.6802 | 0.6418 | 0.1507 | 0.074% | 300,000 |
| XGBoost (independent horizons) | 0.7121 | 0.7085 | **0.7117** | 0.6865 | **0.6523** | 0.1478 | 1.687% | 1,573,299 |
| XGBoost (300k) | **0.7108** | **0.7082** | **0.7114** | 0.6867 | 0.6507 | 0.1483 | 2.219% | 300,000 |
| Cox PH | 0.6987 | 0.6879 | 0.6838 | 0.6838 | 0.6438 | 0.1434 | **0.000%** | 300,000 |
| Unconstrained NN | 0.6879 | 0.6786 | 0.6388 | 0.6598 | 0.6362 | 0.1819 | 97.515% | 300,000 |
| Soft-penalty NN | 0.7030 | 0.6905 | 0.6850 | 0.6881 | 0.6491 | **0.1397** | 0.002% | 300,000 |
| Monotone-architecture NN | 0.7038 | 0.6920 | 0.6871 | **0.6883** | 0.6482 | 0.1429 | **0.000%** | 300,000 |

### The two C-indices disagree, and both are reported

Harrell's C puts the monotone network first at 0.6883 against XGBoost's 0.6867.
Uno's C puts XGBoost first at 0.6507 against 0.6482. The ordering flips depending
on which you read, so both are in the table.

The disagreement has a cause worth understanding rather than averaging away.
Harrell's C scores the entire survival curve, and the survival arms produce one
continuously in time, while XGBoost produces estimates at exactly three horizons
that the evaluation harness holds flat between and beyond them. That flatness is
a property of the adapter, not of gradient boosting, and it costs XGBoost on a
whole-curve metric. Uno's C reweights by the inverse censoring distribution,
which shifts weight toward the horizons where XGBoost was actually fitted.

Neither is wrong. They measure different things, and a study that reported only
the favourable one would be hiding the shape of its own result.

---

## 3. The central result

### (a) Against an unconstrained network, monotonicity is free

Tightening the constraint improves every metric simultaneously. There is no
trade-off on this axis at all:

| | Unconstrained | Soft penalty (α = 0.1) | Monotone architecture |
|---|---|---|---|
| AUC 12m | 0.6879 | 0.7030 | **0.7038** |
| AUC 24m | 0.6786 | 0.6905 | **0.6920** |
| AUC 36m | 0.6388 | 0.6850 | **0.6871** |
| C-Harrell | 0.6598 | 0.6881 | **0.6883** |
| C-Uno | 0.6362 | **0.6491** | 0.6482 |
| IBS 1-60m | 0.1819 | **0.1397** | 0.1429 |
| PD inversions | 97.515% | 0.002% | **0.000%** |

Constraining the model did not cost discrimination; it bought some. An
unconstrained hazard network spends capacity fitting shapes that are not
survival curves. Removing that freedom is not a handicap, it is a prior that
happens to be true.

### (b) Against per-horizon XGBoost, monotonicity is not free

At matched 300,000-row training size, so the comparison carries no data
advantage:

| Horizon | XGBoost (300k) | Monotone NN | Cost |
|---|---|---|---|
| 12m | 0.7108 | 0.7038 | **0.0070** |
| 24m | 0.7082 | 0.6920 | **0.0162** |
| 36m | 0.7114 | 0.6871 | **0.0243** |

**What that buys: impossible term structures for 14,964 of 674,272 held-out
borrowers drop to exactly 0.**

The cost widens with horizon, from 0.0070 at 12 months to 0.0243 at 36. That is
where the constraint binds hardest, because that is where a term structure has
the most room to misbehave. Whether 0.0243 AUC is worth 14,964 coherent term
structures is a decision for whoever owns the model. It is not a decision the
model should make silently, which is why both columns are in the table.

### One honest complication

The soft-penalty arm, not the monotone architecture, holds the best integrated
Brier score in the study at 0.1397 against 0.1429. IBS measures calibrated
accuracy over the whole 1-to-60-month range, not ranking. On that metric the
structural guarantee costs a little, and the twelve inverted borrowers the soft
penalty leaves behind buy it slightly better-calibrated curves everywhere else.

---

## 4. A penalty is not a guarantee

Both constrained networks report **0.0% violations** on the 1,000-borrower
monotonicity grid. On that measurement they are indistinguishable.

On the full 674,272-row test set they are not:

| Arm | Grid audit (1,000 borrowers) | Full test set | Largest reversal |
|---|---|---|---|
| Soft-penalty NN (α = 0.1) | 0.0% | **12 borrowers** (0.0018%) | 8.05e-06 |
| Monotone-architecture NN | 0.0% | **0 borrowers** (0.000%) | 0.0 |

A violation rate near 1 in 56,000 is invisible to a 1,000-borrower sample. It is
not invisible to a loan book.

The soft penalty adds `α · mean[relu(-∂Λ/∂t)²]` to the loss. That makes a
violation *expensive*. It does not make it *impossible*: the penalty is evaluated
at finitely many sampled collocation points, it competes against the data
likelihood, and it is satisfied only to the degree the optimiser finds
worthwhile. Between collocation points, and anywhere the likelihood pulls hard
enough, the curve still turns the wrong way.

The monotone architecture does not have a penalty term to tune. It outputs the
instantaneous hazard through a Softplus and integrates it:

```
λ(x, t) = softplus(f(x, t)) > 0        # strictly positive for any parameters
Λ(x, t) = ∫₀ᵗ λ(x, s) ds               # so this cannot decrease
S(x, t) = exp(-Λ(x, t))                # so this cannot increase
```

The integral of a positive function does not decrease. There is no weight to
choose, no collocation grid, and nothing to trade against the likelihood. The
residual `relu(-dΛ/dt)²` is still computed once as a diagnostic and evaluates to
exactly `0.0`, which checks the integration rather than the fit.

**For a regulatory model this distinction is the whole argument.** "We penalise
violations and observe very few" is a claim about an optimiser's behaviour on a
sample, and it has to be re-validated on every refit, every new segment and every
population shift. "Violations cannot occur" is a property of the function class,
and it survives all of them.

---

## 5. Why not just use Cox

Cox proportional hazards gets monotonicity for free and is what the industry
already uses. It factorises the hazard into a baseline that depends only on time
and a multiplier that depends only on covariates:

```
λ(x, t) = λ₀(t) · exp(x'β)    ⟹    Λ(x, t) = Λ₀(t) · exp(x'β)
```

The Breslow baseline is non-decreasing and `exp(x'β)` is positive, so the product
cannot decrease. Cox scores exactly 0 violations here, as it must.

The price is the proportional hazards assumption: the *ratio* between two
borrowers' hazards is constant for the life of the loan. A borrower twice as
risky at month 6 must still be exactly twice as risky at month 48.

**It is false in this data, and it fails hardest on the strongest predictors.**
From [`results/nb05_ph_assumption_test.csv`](results/nb05_ph_assumption_test.csv),
scaled Schoenfeld residuals with a rank time transform, 5 of 14 covariates
violate at p < 0.05:

| Covariate | χ² | p |
|---|---|---|
| **int_rate** | **18.38** | **1.8e-05** |
| **grade** | **12.84** | **3.4e-04** |
| emp_length | 6.40 | 0.0114 |
| term | 5.87 | 0.0154 |
| emp_length_missing | 4.15 | 0.0415 |

Holding PH: `annual_inc`, `dti`, `loan_amnt`, and all six `home_ownership` levels.

The two that break the assumption by an order of magnitude more than anything
else are Lending Club's own underwriting judgement. Their effect on the hazard
drifts as the loan seasons, which is what you would expect if underwriting is
sharpest about early default and decays afterwards.

**This is why the monotone architecture has a reason to exist.** Cox and the
monotone network both deliver a structurally valid term structure. Only one of
them does it without assuming something the data rejects for its two most
important variables, because in the monotone network λ depends on x and t jointly
and a covariate's effect is free to change shape over time.

---

## 6. Most of the measured skill is borrowed

`int_rate` is not an independent observation about a borrower. It is Lending
Club's own risk assessment written down as a number, and `grade` is the same
judgement on a letter scale. A model that scores well using them is partly
re-deriving the lender's pricing decision rather than predicting default from
borrower fundamentals.

From [`results/int_rate_ablation.csv`](results/int_rate_ablation.csv), both models
refitted on identical rows with `int_rate` and `grade` removed:

| Model | 12m | 24m | 36m |
|---|---|---|---|
| XGBoost | 0.7108 → 0.6120 (**−0.0988**) | 0.7082 → 0.6143 (**−0.0939**) | 0.7114 → 0.6205 (**−0.0909**) |
| Monotone NN | 0.7038 → 0.6068 (**−0.0970**) | 0.6920 → 0.6044 (**−0.0876**) | 0.6871 → 0.6068 (**−0.0804**) |

Both collapse to roughly 0.61.

Chance is 0.50. XGBoost's headline 0.7108 is 0.2108 above chance, and 0.0988 of
that disappears when two columns are removed. **Close to half the discriminative
power above chance is the model re-learning Lending Club's risk pricing, not
predicting default from borrower characteristics.**

This compounds with Section 5: the covariates carrying most of the signal are the
same ones whose effect provably drifts over the life of the loan.

---

## 7. Training size is not the explanation, but it does affect coherence

The survival arms train on 300,000 rows. To check that no classical lead is just
a data advantage, both classical models were refitted on the identical stratified
draw. From [`results/nb02_training_size_comparison.csv`](results/nb02_training_size_comparison.csv):

| Model | 12m | 24m | 36m |
|---|---|---|---|
| XGBoost, 1,573,299 → 300,000 | −0.0013 | −0.0003 | −0.0003 |
| Logistic, 1,573,299 → 300,000 | −0.0002 | −0.0002 | −0.0001 |

Cutting the training data by more than five times costs XGBoost at most 0.0013
AUC and Logistic at most 0.0002. Both are saturated well below 300,000 rows,
which is unsurprising for 14 columns and 200 depth-4 trees. **The gap in
Section 3(b) is real discrimination, not a data advantage.**

Monotonicity, however, gets *worse* at the smaller size, in both arms
independently:

| Model | PD inversions at 1.57M | at 300k |
|---|---|---|
| XGBoost | 1.687% | **2.219%** |
| Logistic | 0.046% | **0.074%** |

Two unrelated model families moving the same direction supports the reading that
less data produces a less coherent term structure, even where it barely touches
discrimination. Accuracy saturates long before consistency does.

---

## 8. Reproduction

`data/` is gitignored. Download **`accepted_2007_to_2018Q4.csv.gz`** from the
[Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(2,260,701 rows, 151 columns) and place it in `data/`.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Notebooks/01_lendingclub_data_preparation.ipynb
```

Notebook 01 must run first; it writes the cleaned CSV every other notebook reads.
Then run 02 through 06 in order. Notebook 06 assembles
`results/master_comparison.csv` from the per-arm JSONs.

### Attrition ledger

Every filtering step is recorded in
[`results/nb01_attrition.csv`](results/nb01_attrition.csv), so the row count is
auditable rather than asserted:

| Step | Rows | Dropped | Retained |
|---|---|---|---|
| Loaded from raw Kaggle file | 2,260,701 | — | 100.000% |
| `dropna(issue_d, last_pymnt_d)` | 2,258,241 | 2,460 | 99.891% |
| `time > 0` | 2,249,276 | 8,965 | 99.495% |
| `dropna(modelling columns)` | **2,247,571** | 1,705 | 99.419% |

### Layout

```
src/common.py        split, features, subsample - single source of truth
src/evaluate.py      C-indices, time-dependent AUC, IBS, calibration
src/monotonicity.py  the violation audit every arm runs identically
src/torch_arms.py    shared training loop for the three neural arms
src/report.py        builds master_comparison.csv from the per-arm JSONs
Notebooks/01..06     data prep, then one notebook per arm family
results/             every number in this README, committed
```

Every arm writes `results/<arm>.json` with `metrics`, `monotonicity` and `notes`
blocks. The master table is generated by reading those files, never by
transcription, so a stale arm appears as a missing row rather than a plausible
wrong number.

---

## 9. Known limitations

**Censoring is informative, and this biases every survival estimate.** `time` is
derived from `last_pymnt_d`, so a Fully Paid loan is treated as censored at its
payoff date. Payoff is not censoring. It is a competing risk: a loan that is paid
off cannot subsequently default, and prepayment correlates with creditworthiness,
so the borrowers leaving the risk set are systematically better than those
remaining. Every survival curve here is therefore biased. For `Current` loans the
correct censoring time is the snapshot date minus the issue date, not the last
payment date, which censors them too early. Fixing this requires a competing-risks
formulation with separate cause-specific hazards for default and prepayment, and
a snapshot-date-based censoring time for open loans. Nothing in this comparison
corrects it, and no absolute PD here should be read as calibrated. The biases push
every arm the same direction, so the ranking across arms survives; the levels do
not.

**Survival arms train on 300,000 rows, classical arms have full-data rows too.**
Cox and the three neural arms do not finish in acceptable wall-clock on all
1,573,299 rows on this machine. The matched-size classical rows exist so the
headline comparison in Section 3(b) carries no confound; the full-data rows are
the best those models do here. Every arm records its own `n_train`.

**Cox carries a ridge penalizer of 0.1.** The shared design matrix one-hot
encodes `home_ownership` without dropping a level, so those columns are exactly
linearly dependent and the Cox information matrix is singular without it. The
alternative was giving Cox a different feature matrix from every other arm, which
would break the comparison. Recorded in the arm's `notes` block.

**Two neural arms hit the epoch ceiling instead of converging.** The unconstrained
NN and the soft-penalty NN both stopped at `best_epoch: 56` of a 60-epoch cap with
`stopped_early: false`, meaning validation loss was still improving when training
was cut off. Their numbers are floors, not estimates. The monotone architecture
converged properly, stopping early at epoch 30 of 35 run. Raising the ceiling
would move two of the three neural rows and not the third, so the neural arms are
not perfectly matched on optimisation budget.

**Eight source features, 14 columns.** `annual_inc`, `dti`, `loan_amnt`,
`int_rate`, `term`, `grade`, `emp_length`, `home_ownership`. No FICO, no
`revol_util`, no `purpose`, no `sub_grade`. All are present in the raw Kaggle file
and absent from the cleaned intermediate this study reads, so adding them means
re-running notebook 01 with a wider column set.

**Uno's C is computed by the local harness, not scikit-survival.** `lifelines` is
available in this environment and `scikit-survival` is not, recorded in each arm's
`notes.backends`.

---

## What would change the conclusion

The competing-risks fix in the first limitation is the one that matters. Treating
payoff as censoring biases every curve in the study, and it is the difference
between a model that ranks borrowers and a model a regulator would accept. That
is a change to notebook 01's outcome definition and a cause-specific hazard per
arm, not a tuning exercise.
