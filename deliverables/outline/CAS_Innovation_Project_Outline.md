# Regularized GLM–Neural Network Hybrids for Insurance Pricing

*CAS ETH in Machine Learning in Finance and Insurance — Innovation Project Outline*

> *Note: this document was drafted with AI assistance (Claude).*

## Motivation & Topic

Wüthrich et al.'s *AI Tools for Actuaries* [1] presents two complementary ways to combine GLMs with neural networks: the Combined Actuarial Neural Network (CANN), which freezes the GLM and adds a correction network as a skip connection, and LocalGLMnet, which lets a network learn locally-varying GLM coefficients.
Both preserve interpretability, but via different architectural choices.
This project explores a third option: training the GLM and a small correction network jointly, with the correction branch (initialized near zero) shrunk toward the pure-GLM solution via an explicit regularization penalty (ridge, with LASSO as an optional extension).
Regularization strength becomes a tunable dial between GLM-like interpretability and FNN-like flexibility, rather than a hard architectural constraint.

## Objective & Contribution

Rather than comparing model variants on predictive performance alone, the project assesses whether a plain GLM structure should be preferred unless additional nonlinear structure yields a sufficient improvement in likelihood, and whether asymmetric regularization can encode this preference within a single jointly-trained architecture.
Predictive performance is evaluated via out-of-sample Poisson deviance (see Evaluation).

Departure from GLM structure is assessed using the per-covariate SHAP dependence plot diagnostic from Mayer, Meier & Wüthrich [4], comparing each covariate's implied shape to the GLM's own linear term for it.
These dependence plots serve as an explainability diagnostic, without treating distance from the GLM as a numeric measure of explainability itself.

The central result is how predictive performance and departure from GLM structure move together as λ varies, illustrated by dependence plots at representative values.

## Method & Data

### Model specification

$$
Y_i \mid X_i = x_i \sim \operatorname{Poisson}(\mu_i), \qquad \log \mu_i = \log e_i + \eta(x_i),
$$

where $e_i$ denotes exposure (entering as an offset, so $\mu_i = e_i \cdot \exp(\eta(x_i))$, consistent with the frequency-as-rate setup in the source paper).
The following specifications of the predictor $\eta$ are compared (XGBoost is included as an additional standard tabular-ML benchmark if implementation time permits; no extensive hyperparameter optimization is planned):

$$
\begin{aligned}
\text{GLM:}\qquad & \eta(x) = x^\top \beta, \\[4pt]
\text{XGBoost:}\qquad & \eta(x) = f_{\mathrm{XGB}}(x), \\[4pt]
\text{FNN:}\qquad & \eta(x) = f_\theta(x), \\[4pt]
\text{CANN:}\qquad & \eta(x) = x^\top\beta + g_\theta(x), \\[4pt]
\text{GLM-favouring FNN:}\qquad & \eta(x) = h_{\beta,\theta}(x),
\end{aligned}
$$

where $h_{\beta,\theta}$ is a neural network whose parameterization and/or regularization favours the GLM-like parameters $\beta$ over the general neural-network parameters $\theta$.
Concretely, this is the same skip-connection form as CANN,

$$
h_{\beta,\theta}(x) = x^\top\beta + g_\theta(x),
$$

but where CANN fixes $\hat\beta = \hat\beta_{\mathrm{GLM}}$ and trains only $\theta$, the proposed model trains $\beta$ and $\theta$ **jointly**, with an asymmetric penalty on $\theta$:

$$
(\hat\beta, \hat\theta) = \arg\min_{\beta,\theta} \left\{ D_{\mathrm{Poisson}}(\beta,\theta) + \lambda \|\theta\|_2^2 \right\}.
$$

("$\arg\min$" here is not literal: gradient-based FNN fitting is non-convex and typically converges to a local, not global, minimum — as for the FNN and CANN benchmarks above.)

This is the project's main contribution, replacing CANN's hard freeze on $\beta$ with a soft, tunable preference for GLM-like behaviour.

- **GLM.** $\eta(x) = x^\top\beta$, fit by maximum likelihood (standard IRLS/MLE).
  The interpretable baseline and the reference coefficients against which every other variant's SHAP values are checked.
- **FNN.** $\eta(x) = f_\theta(x)$, a small feed-forward network (2–3 hidden layers) trained directly on the Poisson deviance loss.
  The unconstrained neural-network benchmark and the natural least-interpretable reference point — but not necessarily the best performer, since it must learn the GLM's structure (e.g. the log-linear main effects) from scratch rather than getting it for free, which can hurt generalization with limited data.
  This is part of the motivation for CANN and the GLM-favouring FNN below.
- **XGBoost (optional).** $\eta(x) = f_{\mathrm{XGB}}(x)$, a gradient-boosted tree ensemble trained on the same Poisson deviance loss.
  Conceptually plays the same role as the FNN in this comparison — a flexible, non-parametric model that can fit interactions and non-linearities without them being specified, with weak native explainability — just via a different function class (tree splits rather than a network).
  A potential strong reference point (standard, well-tuned tabular baselines are often hard to beat, and it preempts the question of why only neural architectures are being compared against the GLM); included as a possible additional baseline if time allows, with default hyperparameters only — not required, and not a target for tuning effort, for the core comparison.
- **CANN (paper baseline).** $\eta(x) = x^\top\beta + g_\theta(x)$.
  The GLM part $\beta$ is fit first and frozen; the correction network $g_\theta$ is added as a skip-connection branch and trained afterwards on the residual structure the GLM misses, exactly as described in §5.6 of the source paper.
- **GLM-favouring FNN (main contribution).** $\eta(x) = h_{\beta,\theta}(x)$, same skip-connection architecture as CANN, but $\beta$ and $\theta$ are trained jointly in a single pass (see loss above): $\theta$ is initialized near zero (identity-style initialization, as in ResNet) and shrunk toward zero unless the deviance-loss improvement justifies it.
  The regularization strength $\lambda$ becomes the tunable dial between GLM-like interpretability ($\lambda \to \infty \Rightarrow \theta \to 0 \Rightarrow \eta(x) \approx x^\top\beta$) and an increasingly flexible jointly-trained GLM+NN as $\lambda \to 0$ — note this is *not* literally equivalent to the standalone FNN above, since the GLM skip connection remains part of the architecture regardless of $\lambda$; this replaces CANN's hard architectural freeze with a soft, continuously adjustable constraint.

### Data

- **Primary: French MTPL claim frequency data.** Used for direct comparability with the results reported in the source paper — reusing an established dataset (and the paper's own published baseline numbers) means the GLM, frozen CANN, and regularized-variant pipeline can be up and running faster, without first having to design and validate a data-generating process.
  This is the dataset the core comparison and the trade-off curve are built on.
- **Secondary (optional extension): synthetic frequency dataset.** Covariates drawn independently (uniform/normal, realistic ranges akin to age or vehicle power).
  True mean frequency function combines (i) a linear part in 2–3 covariates — the part a plain GLM should fully recover; (ii) one deliberate non-linearity (e.g. a quadratic or threshold effect) in another covariate; and (iii) one genuine interaction term between two covariates, which a GLM without explicit interaction terms cannot capture.
  Claim counts are simulated as independent Poisson draws with mean v·μ(X), where exposure v varies across policies — this changes the distribution per policy (no longer identically distributed) but not the independence between policies.
  If time allows after the MTPL results are in hand, this dataset adds an exact-ground-truth recovery check (does each model variant, and its SHAP decomposition, recover the true linear, non-linear, and interaction effects?) on top of the MTPL-based trade-off curve.
  **Note:** since the true decomposition is known here, this is also the only dataset that could actually test the identifiability question raised above (§ Model specification note) — MTPL alone can establish predictive performance and departure-from-GLM behaviour, but not whether the model recovers the "right" structure.
  This makes the synthetic dataset more valuable than a simple robustness check, though it remains optional given the timeline; if pursued, it would be worth prioritizing over polishing additional MTPL diagnostics.
- **Tertiary (out of scope): employer's health insurance claim frequency data.** A proprietary dataset from the employer, noted here as a natural real-world extension beyond MTPL, but explicitly excluded from this iteration given the project's time constraints (data access/preprocessing overhead, and confidentiality considerations — results would need to be presented in aggregated form per the CAS guidelines).

### Training & evaluation details

- All models trained on the Poisson deviance loss (the maximum-likelihood loss implied by the Poisson assumption), so comparisons are apples-to-apples across GLM, FNN, and hybrid variants.
- Regularization strength λ swept over a grid to trace out the performance–structure trade-off (see Evaluation, below), rather than tuned to a single "best" value.
- SHAP values computed per model: closed-form for the pure GLM (reduces to coefficient × centered covariate), and standard SHAP estimators (as covered in the source paper's §9.5.3) for the FNN-containing variants.
- **GLM feature scope.** The GLM uses the same, full feature set as the correction network (the established MTPL feature list and standard transformations from the literature), rather than a hand-tuned or significance-reduced subset.
- **Data split.** A single fixed split, 56% train / 14% validation / 30% test, drawn once (fixed random seed) and reused identically across every model variant and every point on the λ sweep — otherwise variation in the split would leak into the trade-off curve and make model/λ differences hard to distinguish from split noise.

### Scope

The analysis is restricted to claim frequency (Poisson) modelling; severity modelling, additional datasets (e.g. proprietary employer data), and the group-LASSO variant are noted as natural extensions but are out of scope for this iteration.
The core comparison runs on the French MTPL data; the synthetic ground-truth recovery check is the first item to trim if time is tighter than planned.

### Out of scope

- Employer's health insurance claim frequency data (see Tertiary dataset, above).
- A gamma severity model (claim size, given a claim occurred) — this project restricts itself to claim frequency (Poisson); combining frequency and severity is a natural follow-up but adds a second full set of model variants and is not pursued here.
- Reproducing LocalGLMnet (§5.7 of the source paper) as a fifth model variant — it's a genuinely different way of keeping explainability built in (locally-varying GLM coefficients, rather than a skip-connection correction term), but implementing and tuning it as its own baseline is a separate effort from this project's core regularization contribution.

## Literature Review (TODO — mini review, optional per guidelines)

A short review situating the project relative to the source paper and its underlying references.
Roles: [1] is the main textbook-style resource for this project; [2] serves as a comprehensive reference to dip into as needed rather than read cover-to-cover; [3] and [5] are important "cultural"/foundational reads that establish the shared vocabulary (CANN, SHAP) the rest of the field builds on; [4] plays the same foundational role specifically for the SHAP-in-actuarial-pricing angle.
To be written up, covering at least:

1. Wüthrich, M.V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J. & Scognamiglio, S. — *AI Tools for Actuaries* — **main textbook-style resource** for this project.
   Recommended: Ch. 3 *Generalized linear models*, pp. 57–70 (~14 pp); Ch. 5 *Feed-forward neural networks*, pp. 89–116, incl. §5.6 *Combing a GLM and a neural network* (~2 pp) and §5.7 *LocalGLMnet* (~27 pp total); Ch. 9 *Explainability*, pp. 189–220, esp. §9.5 SHAP, pp. 207–218 (~31 pp / ~11 pp for §9.5 alone).
   Rough total for the recommended subset: ~70 pages.
2. Wüthrich, M.V. & Merz, M. — *Statistical Foundations of Actuarial Learning and its Applications* — **comprehensive reference**, to consult rather than read end-to-end.
   Recommended chapters: Ch. 2 *Exponential Dispersion Family*, pp. 13–47 (~34 pp); Ch. 5 *Generalized Linear Models*, pp. 111–205 (~94 pp); Ch. 6 *Bayesian Methods, Regularization and Expectation-Maximization*, pp. 207–266 (~59 pp); Ch. 7 *Deep Learning*, pp. 267–379 (~112 pp).
   Rough total for the recommended subset: ~300 pages — likely too much for this project's scope; a lighter pass would prioritize Ch. 2 and the regularization sections of Ch. 6.
3. Schelldorfer, J. & Wüthrich, M.V. — *Nesting Classical Actuarial Models into Neural Networks* — **important foundational read**: the original CANN paper and the basis for the field's shared vocabulary around the frozen-GLM baseline being compared against.
   Rough estimate: ~30 pages (typical length for this SSRN tutorial series).
4. Mayer, M., Meier, D. & Wüthrich, M.V. (2023) — *SHAP for Actuaries: Explain any Model*, SSRN Manuscript ID 4389797 — **foundational read** for the explainability diagnostic used in this project (per-covariate SHAP dependence plots vs. GLM structure).
   Rough estimate: ~25–30 pages (typical length for this actuarial tutorial series; to be confirmed on first read).
5. Lundberg, S.M. & Lee, S.-I. (2017) — *A Unified Approach to Interpreting Model Predictions*, NeurIPS — **important foundational read** establishing the SHAP framework itself, "covering the base" before the actuarial-specific applications in [1] and [4].
   ~10 pages (NeurIPS proceedings, pp. 4765–4774) + supplementary appendix.

## Links / Reminders

- Statistical Foundations of Actuarial Learning and its Applications (Moodle copy): https://moodle-app2.let.ethz.ch/pluginfile.php/2528629/mod_folder/content/0/Statistical%20Foundations%20of%20Actuarial%20Learning%20and%20its%20Applications.pdf?forcedownload=1
- AI Tools for Actuaries (SSRN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304

## Evaluation

- Predictive performance: Poisson deviance, Gini index, lift curves (as in the paper).
- Departure from GLM structure: SHAP-based deviation from the GLM baseline (not treated as "explainability" itself — see Objective & Contribution); recovery of true effects on synthetic data if pursued; sensitivity of both metrics to regularization strength.

## Real-World Applicability

A regularization-controlled interpretability dial offers actuaries a practical governance lever: the same architecture can be tuned toward strict auditability or toward maximal predictive lift, with the trade-off made explicit and measurable rather than fixed by architecture alone — directly relevant to model governance and regulatory review of pricing models.

## Work Timeline (Aug 25 – Nov 7, 2026)

**Note:** Personal target dates are set ahead of the official CAS deadlines (outline officially due Sep 15, report officially due Oct 31) to leave a buffer.
Only **limited project work** is possible from **Sep 26 – Oct 11** (holiday); work before the break is front-loaded to get the French MTPL pipeline (the core comparison) running and hit the Sep 10 outline target.
The post-break window (Oct 12–29) carries the synthetic-data extension (if time allows), report writing, and buffer in a compressed ~2.5 weeks — to make this realistic, plan to take additional time off from regular work during this Oct 12–30 period, beyond the holiday itself, to free up dedicated project time.

**Effort budget:** at a realistic ~15 h/week, the ~9.4-week span (Aug 25 – Oct 29) minus the lighter holiday stretch works out to roughly 105–110 hours available before counting any extra days off — comfortably above the standard ETH estimate of ~75–90 hours for a 3 ECTS course (25–30 h/credit; not stated explicitly in the guideline, treat as a rough reference point only).
The plan has margin built in, even without the extra time off from regular work mentioned above.

| Period | Milestone / Activity |
|---|---|
| Aug 25 – Aug 31 | Re-read relevant chapters of *AI Tools for Actuaries* (GLM, FNN, CANN, LocalGLMnet, explainability/SHAP); draft mini-literature review notes on references [1]–[5]; finalize project scope; draft 1-page outline. |
| Sep 1 – Sep 7 | Refine outline, modelling plan, and literature review notes. Start MTPL pipeline groundwork: set up compute environment (Python env, GLM/NN libraries, GPU access if used), download and import the French MTPL dataset, first look at the data (schema, exposure field, basic checks). |
| Sep 8 – Sep 9 | Finalize outline. |
| **Sep 10, Thu (deadline)** | **Submit 1-page outline to Bastian Bergmann for approval.** *(personal target; official deadline Sep 15)* |
| Sep 10 – Sep 21 | Continue MTPL pipeline: preprocessing/feature setup; implement pure GLM and frozen CANN baselines; implement jointly-trained, ridge-regularized GLM+FNN variant (skip-connection architecture, near-zero-initialized correction branch); run first predictive-performance comparison (Poisson deviance, Gini/lift). |
| Sep 22 – Sep 25 | Run SHAP on the MTPL-based models; draft first performance–structure trade-off curve. Wrap up and leave clean notes before the break. |
| **Sep 26 – Oct 11** | **Holiday — limited project work.** |
| Oct 12 – Oct 18 | Resume: finalize MTPL results and trade-off curve. If time allows, build the synthetic dataset (optional extension) for an exact-ground-truth recovery check. |
| Oct 19 – Oct 23 | Consolidate MTPL results and figures; run the synthetic-data extension if time allows. If time is short, drop the synthetic extension rather than the MTPL results. |
| Oct 24 – Oct 28 | Write and finalize project report (method, experiments, results, discussion of real-world applicability), including the literature review section. |
| **Oct 29, Thu (deadline)** | **Submit final project report.** *(personal target; official deadline Oct 31)* |
| Oct 30 – Nov 5 | Prepare slides and rehearse 15-minute presentation. |
| Nov 6 – Nov 7 | Project presentation (15 min + 5 min Q&A). |

## Methodological Reminder

FNN and correction-network hyperparameters (architecture, learning rate, optimizer) are kept close to those used in the source paper's own FNN/CANN examples, rather than independently searched — consistent with the "don't spend significant tuning effort" boundary applied to XGBoost, and keeping the single validation split (see Data split, above) sufficient for early stopping without needing k-fold CV.

Regularize **all** of $\theta$ (not just the last layer): with ReLU, scaling up an early layer's weights by $a$ and the last layer's by $1/a$ leaves $g_\theta$ unchanged but shrinks a last-layer-only penalty by $1/a^2$ — a free loophole that lets the optimizer dodge the penalty without actually shrinking the correction.
Penalizing all layers avoids this.
(A sigmoid/tanh correction network wouldn't have this loophole, since it isn't positively homogeneous, but trains less well than ReLU.
Alternative: penalize $g_\theta(x)$'s output directly, e.g. $\lambda \cdot \mathrm{mean}(g_\theta(x)^2)$, which sidesteps the issue regardless of activation.)

**GLM feature scope.** Significance-based reduction (dropping insignificant terms via Wald/LR tests) is a standard technique but is deliberately not applied to the GLM here: since both branches need access to the same inputs for the identifiability comparison to be meaningful, and since coefficient significance is itself unreliable under the GLM misspecification this project assumes, letting MLE naturally shrink weak coefficients toward zero is preferred over structurally excluding features.

**Identifiability (not yet adopted as the core framing):** because $g_\theta$ can itself represent linear functions, the decomposition of $\eta$ into $x^\top\beta$ and $g_\theta(x)$ is not identifiable without some preference between the two — which is exactly what the $\lambda\|\theta\|_2^2$ penalty provides, by making it cheaper to route genuinely linear structure through $\beta$.
This suggests a sharper, falsifiable framing of the contribution ("does asymmetric regularization make the joint model preferentially allocate simple effects to the GLM branch, reserving $\theta$ for genuine residual nonlinearity?") as an alternative or addition to the accuracy/departure trade-off framing above.
Flagged here as a promising direction, not yet committed to — would need the synthetic experiment (see Data) to actually test it, since only there is the true decomposition known.
