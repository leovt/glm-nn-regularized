# Presentation notes

Notes accompanying [presentation.tex](presentation.tex).

## 1. Title slide

## 2. Research question

## 3. Regularized mixed GLM-FNN data flow

## 4. GLM data flow

## 5. Input preprocessing for the linear and neural branches.

## 6. Implementation challenges

## 7. GLM in Keras

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

Source: [CANN teaching notebook](https://github.com/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN.ipynb);
project results and interpretation: [report](../report/report.tex).

## 8. FNN data flow

## 9. CANN data flow

## 10. Regularized mixed GLM-FNN data flow

## 11. Selected saved notebook results

## 12. Joint-hybrid sweep with batch size 50,000 and 200 epochs.

## 13. Regularization path of the jointly trained hybrid

## 14. GLM parameters for selected features for different regularization strengths

## 15. FNN correction in regularized model

## 16. FNN correction in regularized model - Histograms

## 17. Real-world applicability

## 18. Lessons learned

## 19. Appendix

## 20. GLM in Keras
