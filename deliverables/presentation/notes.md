# Presentation notes

Notes accompanying [presentation.tex](presentation.tex).

## 1. Title slide
### Timing
| Slides | Topic                                                   | Time  |
| ------ | ------------------------------------------------------- | ----- |
| 1–5    | Research question, architecture, GLM and preprocessing  | 3:00  |
| 6–7    | Implementation challenges and Keras GLM                 | 2:00  |
| 8–11   | FNN, CANN, joint hybrid and benchmarks                  | 2:30  |
| 12–15  | Main results: regularization path and model diagnostics | 5:00  |
| 16–17  | Real-world applicability and lessons learned            | 1:30  |
|        | Total planned speaking time                             | 14:00 |
|        | Presentation limit                                      | 15:00 |

### Key Points

2. CANN → joint training
3. Target architecture · Explain later
4. Three stages: preprocess → predict → transform
5. Expert judgement
6. Keras vs IRLS (why) · No FNN final bias
7. Batch size / validation split · Calibration error
8. Embeddings · AI Tools for Actuaries
9. AI Tools for Actuaries · statsmodels GLM
10. Reprise of slide 3 - Regularization
11. Batch size
12. Transition
13. GLM betas · Driver-age dependency
14. Intermediate-output extraction
15. Simplification
16. Risk model ≠ commercial tariff
17. Selection measures · Regularization-path transition

## 2. Research question

Start with the short answer: “I can control the neural contribution,
but not smoothly, and I did not show improved interpretability.”

The research question is whether jointly training a GLM and a regularized
neural correction can preserve predictive performance while favouring
a more interpretable structure. I trained both branches jointly,
penalising the neural dense-layer weights while leaving the GLM
unpenalised, then examined predictive fit, GLM coefficients and the neural
contribution along the regularization path.

## 3. Regularized mixed GLM-FNN data flow
Tell the audience this is the architecture I'll return to after introducing its components.

## 4. GLM data flow
A GLM is not just a mathematical model; its practical implementation incorporates expert judgement.

## 5. Input preprocessing for the linear and neural branches.
The GLM's structure reflects expert decisions made before fitting.
The FNN reduces some of that manual feature engineering, but does not eliminate modelling choices.

## 6. Implementation challenges
**Presenter cue: Should have 12 minutes left**

## 7. GLM in Keras
**Presenter cue: Avoid spending too long on slide 7**

Both the project implementation and the CANN teaching notebook use
`optimizer="nadam"` without specifying a learning rate, so both use
Nadam's default learning rate of 0.001
([Keras documentation](https://keras.io/api/optimizers/Nadam/)).

The batch size of 5,000 was tried following the CANN example in
*AI Tools for Actuaries*. With a 90/10 fitting/validation split,
these runs showed a considerable aggregate calibration error:
test observed-to-expected claims ratios were 1.0313 for the randomly
initialised model and 1.0367 for the stability test. These correspond
to approximately 3.1 and 3.7 percent more observed than expected claims.

Test Poisson deviance nevertheless remained similar to the Keras
control: 47.2504 and 47.1970, compared with 47.1786. Predictive fit
and aggregate calibration therefore need separate assessment.
The validation split also changes the fitting sample, so its optimum
need not equal the full-learning-sample maximum likelihood estimate.

Learning and test deviances are also similar: 47.0045 versus 47.2504
for the randomly initialised model with validation, and 46.9668 versus
47.1970 for the stability test. The control shows a comparable gap
(46.9537 versus 47.1786). These results do not suggest a large
generalisation gap in these runs, although they do not establish the
absence of overfitting or stability across samples and random seeds.
The reported learning scores include the validation observations.

Source: [CANN teaching notebook](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN.ipynb);
project results and interpretation: [report](../report/report.tex).

## 8. FNN data flow
**Presenter cue: Should have 10 minutes left**

The FNN implementation is taken from the *AI Tools for Actuaries*
[source material](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN.ipynb).

One-hot encoding represents a category by a vector with one entry per
category: one entry is one and all others are zero. An embedding instead
assigns each category a short vector whose values are learned jointly
with the model. Here, vehicle brand and region each use two-dimensional
embeddings. Categories can acquire similar representations if this helps
predict claim frequency; their numerical category codes imply no ordering.
An embedding lookup is equivalent to multiplying a one-hot vector by a
learned matrix, but avoids constructing that sparse vector explicitly.

## 9. CANN data flow

The CANN implementation is taken from the *AI Tools for Actuaries*
[source material](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN.ipynb).

## 10. Regularized mixed GLM-FNN data flow

The jointly trained hybrid combines the GLM linear predictor with a neural correction on the log-frequency scale:

\[
\log \hat{\mu}(x)=z(x)^\top\beta+g_\theta(u(x)).
\]

Unlike the frozen CANN, both branches are trainable. However, we apply an **asymmetric regularization penalty**: the GLM coefficients remain unpenalized, while the neural network's dense-layer weights are penalized.

The training objective is the Poisson loss plus an L2 penalty:

\[
\mathcal L(\beta,\theta)
=
\mathcal L_{\mathrm{Poisson}}(\beta,\theta)
+
\lambda\sum_{\ell}\|W_\ell\|_F^2.
\]

Here, \(\lambda\) is a **hyperparameter controlling the strength of regularization**. At \(\lambda=0\), we optimize predictive fit without a penalty. As \(\lambda\) increases, large neural weights become increasingly costly. This encourages the neural branch to make smaller or simpler corrections, potentially shifting more of the fitted structure into the GLM.

The penalty does not directly constrain the neural output, nor does it guarantee that the GLM recovers its original coefficients. Both branches can represent overlapping effects, and their interaction is determined by joint optimization. In this implementation, the penalty applies to dense-layer weight matrices, not to the GLM coefficients, embeddings or biases.

The central experiment is therefore to vary \(\lambda\) and observe what happens to predictive deviance, GLM coefficients and the neural correction. This is the regularization path shown in the following results.

## 11. Selected saved notebook results
**Presenter cue: Avoid spending too long on slide 11**

The joint hybrid at \(\lambda=1\) has learning deviance 46.955,
test deviance 47.182 and test O/E 0.9966, illustrating the return to
GLM-like performance under strong regularization.

The GLM scores match the teaching material to the displayed precision.
The neural benchmark scores are close, but not identical:

| Model                    | Project learning / test deviance | Source learning / test deviance |
| ------------------------ | -------------------------------: | ------------------------------: |
| GLM                      |                  46.954 / 47.179 |                 46.954 / 47.179 |
| Frozen CANN              |                  44.970 / 44.955 |                 44.968 / 44.932 |
| FNN                      |                  44.830 / 44.894 |                 44.816 / 44.902 |
| FNN, mean initialisation |                  44.814 / 44.902 |                 44.816 / 44.902 |

The source FNN uses mean initialisation, making the corresponding project
row the closest comparison. Both source notebooks and the project use
neural-network seed 1234, so a different configured seed does not explain
the discrepancies. The effective stopping and weight-selection rules also
match: both run 200 epochs and retain the best validation-loss weights.
The sources use a checkpoint and reload it; the project uses
`EarlyStopping` with patience equal to the epoch budget and
`restore_best_weights=True`, which does not shorten these runs.
Implementation and numerical-environment differences
are possible explanations; their contributions have not been isolated.

The FNN and frozen CANN benchmarks still use batches of 5,000. Their
aggregate calibration errors are consistent with the optimisation drift
seen in the GLM experiments, although this does not establish batch size
as the cause. The larger batch size of 50,000 applies to the jointly
trained hybrid sweep, whose displayed test O/E ratios are closer to one.
The benchmarks' roughly 1–2 percent calibration errors may partly reflect
the batch size of 5,000, given the roughly 3 percent drift observed in
the GLM experiments at that batch size. This is a plausible explanation,
not an established cause.
Batch size and model specification both differ, so their effects cannot
be separated from this comparison alone.

TODO: rerun the FNN and frozen CANN benchmarks with batches of 50,000
and compare deviance and O/E with the existing 5,000-batch runs, keeping
the seed, split, initialisation and best-validation-weight selection fixed.
Check convergence, since larger batches give fewer optimiser updates
per epoch.

Sources: [CANN notebook](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN.ipynb)
and [FNN embedding notebook](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_05_FNNs%20-%20embedding.ipynb).

## 12. Regularization path of the jointly trained hybrid
**Presenter cue: Should have 7 minutes left**

Lambda is the regularization hyperparameter. It controls how strongly we penalize the neural network's weights, while the GLM coefficients remain unpenalized. At lambda zero, there is no penalty; as lambda increases, we increasingly discourage the neural network from contributing to the prediction.

The main experiment investigates how the jointly trained hybrid responds to increasing regularization of the neural network. The GLM coefficients remain unpenalized, while the neural network's dense-layer weights are penalized. We would therefore expect the neural contribution to become less important as regularization increases, potentially allowing the GLM to take over more of the prediction. However, the model is trained jointly, so we cannot assume that the transition will be smooth or that the GLM will recover its original coefficients.

The regularization path shows a relatively stable region at low penalty strengths, followed by a sharp transition around \(\lambda \approx 0.0018\). At this point, the predictive deviance deteriorates toward the GLM level. This is not simply a gradual loss of performance as the penalty increases: the fitted model appears to move between two qualitatively different regimes, one in which the neural branch contributes materially and another in which its contribution is strongly suppressed.

The result illustrates a trade-off between predictive performance and the complexity of the neural component. Relatively small penalties can preserve much of the predictive improvement over the standalone GLM, but stronger penalties eventually eliminate that advantage. The sharp transition is an empirical observation for this architecture and training procedure, not evidence of a universal critical regularization strength. It also raises a question that predictive deviance alone cannot answer: what actually happens to the two components of the hybrid as the penalty increases?

## 13. GLM parameters for selected features for different regularization strengths

To investigate the interaction between the two branches, I extracted selected GLM coefficients from the jointly trained models. Unlike in a frozen CANN, these coefficients are free to change during training. The neural network and GLM can therefore redistribute explanatory effects between them. A change in a GLM coefficient does not necessarily imply a corresponding change in the overall prediction, because the neural branch may compensate for it.

The coefficient paths show that regularization affects more than the magnitude of the neural correction. It also changes the fitted linear component. This is important for the original research question: if we want the hybrid to retain a meaningful GLM structure, we need to examine the coefficients themselves, rather than judging the model solely by predictive deviance. The penalty is applied asymmetrically, but that does not mean that the resulting decomposition is uniquely determined or that the GLM coefficients remain stable.

There is also a limitation to what we can conclude from these paths. A coefficient in the jointly trained GLM branch is not automatically interpretable in the same way as a coefficient from a standalone GLM, because the neural branch may model overlapping effects. The plots provide evidence about how the two components interact under regularization, but they do not establish that the resulting GLM is a more faithful or interpretable representation of the underlying risk relationships.

## 14. FNN correction in regularized model

For this analysis, I extracted the output of the neural correction \(g_\theta(x)\) for individual observations, before it is added to the GLM linear predictor. Since the model uses a log link, this correction is on the log-frequency scale. Exponentiating it gives the multiplicative adjustment applied to the GLM prediction. This allows us to examine the neural branch directly, rather than inferring its contribution only from the final predicted frequencies.

I summarized these corrections using their mean, standard deviation and root mean square. The mean describes the overall level of the correction on the log scale, while the standard deviation describes how much it varies between observations. The RMS combines both effects and provides a measure of the overall magnitude of the correction. In particular, a mean close to zero does not mean that the neural network is unimportant: it could still make substantial positive and negative corrections for different risks.

These statistics help distinguish between a neural branch that mainly shifts the overall prediction level and one that provides substantial differentiation between risks. They also provide a way to track how the neural contribution changes along the regularization path. A correction whose mean and variability both approach zero is becoming negligible in the fitted predictor. However, these are descriptive statistics of the model output, not a direct measure of the accuracy or usefulness of the corrections.

Implementation note: the notebook's `fnn_correction` function creates a
Keras model that exposes an intermediate layer as its output:

```python
correction_model = keras.Model(
    inputs=model_info.model.inputs,
    outputs=model_info.model.get_layer("fnn_linear_predictor").output,
)
corrections = correction_model(
    model_info.train.input, training=False
).numpy().ravel()
```

This model reuses the trained hybrid's graph and layer weights directly;
no separate weight loading or retraining is needed. Its output is the
neural log-rate correction before addition to the linear branch and
exponentiation. `training=False` selects inference behaviour. The notebook
runs one forward pass per fitted hybrid and reuses the resulting arrays
for the mean, standard deviation, RMS and histograms.

## 15. FNN correction in regularized model - Histograms

The histograms show the same neural corrections at selected regularization strengths, but retain information that is lost in the summary statistics. Instead of looking only at the mean or RMS, we can see how the corrections are distributed across observations. A wide distribution indicates that the neural branch applies substantially different multiplicative adjustments to different risks, whereas a distribution concentrated near zero indicates that the model is approaching the GLM prediction.

The selected penalty values illustrate the transition observed in the regularization path. At lower regularization strengths, the neural correction retains a noticeable distribution of values. As the penalty increases toward the transition, this distribution contracts. Around the critical region, a relatively small further increase in regularization produces a much more concentrated distribution, consistent with the sharp deterioration in predictive deviance.

Taken together, the regularization path, coefficient paths and neural-correction diagnostics show that regularization changes both the predictive performance and the internal decomposition of the hybrid. The neural contribution can be strongly reduced, but this does not by itself prove that the remaining GLM component has recovered a uniquely meaningful structure. The experiment therefore demonstrates that asymmetric regularization can control the neural contribution, while leaving open the more difficult question of whether it improves interpretability.

## 16. Real-world applicability
**Presenter cue: 1-2 minutes left**

The frequency model estimates expected claim counts per unit of exposure.
These estimates can inform the risk component of a commercial tariff;
constructing a premium also requires claim severity, expenses and other
pricing considerations beyond this experiment.

Poisson deviance assesses predictive fit, while observed-to-expected
claims ratios assess aggregate calibration. Check calibration both across
the portfolio and within proposed price classes: errors between groups
can cancel in the portfolio total. An intercept adjustment can restore
aggregate balance on a calibration sample, but does not guarantee
subgroup or future calibration.

A commercial tariff may deliberately use fewer, coarser price classes
than the risk model can distinguish. This can make the tariff easier to
explain and maintain while reflecting commercial and operational choices.
The finer risk estimates can help assess expected claims within each
proposed class and the variation concealed by grouping. Check calibration
and risk differentiation again after constructing the tariff; improved
risk estimates do not require a corresponding increase in price classes.
These are potential applications, not tariff-design outcomes tested here.
The practical value of a model depends not only on predictive performance, but also on calibration, stability, interpretability and its usefulness in constructing and maintaining a tariff.

## 17. Lessons learned

Close with three takeaways:

- **Control:** Weak penalties retained predictive gains over the selected
  GLM baseline; stronger penalties suppressed the neural contribution
  and recovered GLM-like performance.
- **Abruptness:** The transition was abrupt in the sampled path, so
  control was not smooth under this training configuration.
- **Fit vs. calibration:** The GLM controls showed calibration drift
  even when deviance remained close to the control. Assess predictive
  fit and calibration separately when selecting models.

The result is control of the neural contribution; improved
interpretability was not demonstrated.

important conclusion on slide 17: the hybrid combines the effort of GLM feature engineering with neural-network tuning, rather than eliminating either.

**Next steps:** Repeat the experiment with multiple seeds, compare models
with comparable feature sets and rerun the FNN and frozen CANN benchmarks
at batch size 50,000. For the batch-size comparison, keep the seed, split,
initialisation and best-validation-weight selection fixed, and check
convergence because larger batches give fewer optimiser updates per epoch.

## 18. Appendix

## 19. GLM in Keras

## 20. Joint-hybrid sweep with batch size 50,000 and 200 epochs.
