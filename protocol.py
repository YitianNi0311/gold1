"""Evaluation protocol shared by the runners and the table builders."""

# Fold 1 is only used to train the first meta model, so stacked models are scored on folds 2-5.
# Every model in the tables is scored on the same folds.
EVAL_FOLDS = (2, 3, 4, 5)
FOLD_SIZE = 736
