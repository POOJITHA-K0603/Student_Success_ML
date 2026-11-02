# Viva notes (understand these, then answer in your own words)

**Why this dataset?** Synthetic, because real student records are private. Documented generator, hidden archetypes, noise and unobserved motivation. Limitation: not real-world proof. Best fix: collect a class survey.

**Target and problem type?** `risk` with 3 classes, so multi-class classification (not regression), because advice depends on category.

**Why Logistic Regression won?** Its CV F1 was within 1 std of the best, with almost no overfitting (train 0.857 vs CV 0.850), fastest, and interpretable. Random Forest/Boosting scored similar but memorised training data (train F1 ~1.0).

**Overfitting: how found, how reduced?** Compared train vs CV/test: unpruned tree 1.00 vs 0.80 test. Validation curve over max_depth. Used simpler models, regularisation (L2 alpha for MLP, early stopping), tuning.

**Metrics?** Accuracy plus macro precision/recall/F1 (all classes matter, At-risk recall especially) and confusion matrix. Silhouette + inertia for clustering.

**Leakage prevention?** Split before everything; imputer/scaler inside Pipeline so they are fitted on train folds only; K-Means/PCA fitted on training data; model chosen using CV, not the test set.

**Why clustering? Why PCA?** No learner-type labels exist; clusters give habit groups for tailored tips. PCA reduces 7 features to 2 for visualising. k=2 had the best silhouette but was too coarse, so k>=3 was searched (disclose this).

**Why a neural network?** To test whether non-linear function learning helps. Result: no gain over Logistic Regression on this data. Know: ReLU, Adam/SGD, backpropagation, L2 penalty, early stopping, learning rate.

**Role of GA?** Search problem: how to split limited hours over 5 topics. Too many combinations to try all; GA evolves allocations using the ML model as fitness. Know selection, crossover, mutation, elitism. Assumption: mastery gain formula with constant 0.12.

**Why not RL?** Allocation for one week is a one-shot optimisation, which suits GA. RL would suit a week-by-week adaptive tutor (future work).

**Tuned parameters?** RF: n_estimators, max_depth, min_samples_leaf (CV chose 300, None, 1). MLP: hidden layers, alpha, learning rate (CV chose (32,16), 0.1, 0.001). See `metrics.json`.

**Limitations?** Synthetic data; GA gain formula assumed; sklearn MLP lacks dropout/batch-norm; no real students validation.
