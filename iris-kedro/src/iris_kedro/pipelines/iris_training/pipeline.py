from kedro.pipeline import Pipeline, node

from .nodes import (
    load_iris_data,
    split_features_target,
    split_train_test,
    train_knn,
    train_logreg,
    train_rf,
    train_svm,
    select_best_model,
    save_model,
    save_metadata,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(load_iris_data, None, "iris_df"),
            node(split_features_target, "iris_df", ["X", "y"]),
            node(
                split_train_test,
                ["X", "y", "params:train_test_split"],
                ["X_train", "X_test", "y_train", "y_test"],
            ),
            node(
                train_knn,
                ["X_train", "X_test", "y_train", "y_test", "params:knn_model"],
                "knn_result",
            ),
            node(
                train_logreg,
                ["X_train", "X_test", "y_train", "y_test", "params:logreg_model"],
                "logreg_result",
            ),
            node(
                train_rf,
                ["X_train", "X_test", "y_train", "y_test", "params:rf_model"],
                "rf_result",
            ),
            node(
                train_svm,
                ["X_train", "X_test", "y_train", "y_test", "params:svm_model"],
                "svm_result",
            ),
            node(
                select_best_model,
                ["knn_result", "logreg_result", "rf_result", "svm_result"],
                ["best_model_name", "best_model"],
            ),
            node(save_model, "best_model", "model_path"),
            node(
                save_metadata,
                ["best_model_name", "best_model"],
                "metadata_path",
            ),
        ]
    )
