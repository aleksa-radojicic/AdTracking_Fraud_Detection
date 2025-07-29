import gc
from time import perf_counter
from typing import Any

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import polars as pl


def do_experiment(
    classifiers: dict[str, tuple[Pipeline, BaseEstimator]],
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for classifier_name, (pipeline, clf) in classifiers.items():
        time_started = perf_counter()
        print(f"Training {classifier_name}...")
        pipeline = clone(pipeline)
        pipeline.steps.append(
            (classifier_name, clf)
        )
        pipeline.set_output(transform='polars')

        pipeline.fit(X_train, y_train)
    
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        y_test_proba  = pipeline.predict_proba(X_test)[:, 1]
    
        auc_train: float = roc_auc_score(y_train, y_train_proba)
        auc_test: float = roc_auc_score(y_test,  y_test_proba)
    
        time_ended = perf_counter()
        time_taken = time_ended - time_started

        _result = {
        "Classifier": classifier_name,
        "AUC (Train)": auc_train,
        "AUC (Test)": auc_test,
        "Time taken": time_taken,
    }
        print(_result)
        print()
        results.append(_result)
        gc.collect()
    return results
