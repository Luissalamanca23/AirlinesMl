"""
Data science pipeline for Airlines ML project.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    prepare_features,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    compare_models
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=prepare_features,
            inputs=["airlines_train", "airlines_val", "airlines_test"],
            outputs=["features_train", "features_val", "features_test", "feature_encoders"],
            name="prepare_features_node",
        ),
        Node(
            func=train_linear_regression,
            inputs=["features_train", "parameters"],
            outputs="linear_regression_model",
            name="train_linear_regression_node",
        ),
        Node(
            func=train_random_forest,
            inputs=["features_train", "parameters"],
            outputs="random_forest_model",
            name="train_random_forest_node",
        ),
        Node(
            func=train_xgboost,
            inputs=["features_train", "features_val", "parameters"],
            outputs="xgboost_model",
            name="train_xgboost_node",
        ),
        Node(
            func=train_lightgbm,
            inputs=["features_train", "features_val", "parameters"],
            outputs="lightgbm_model",
            name="train_lightgbm_node",
        ),
        Node(
            func=evaluate_model,
            inputs=["linear_regression_model", "features_test", "params:model_names.linear_regression"],
            outputs="lr_metrics",
            name="evaluate_lr_node",
        ),
        Node(
            func=evaluate_model,
            inputs=["random_forest_model", "features_test", "params:model_names.random_forest"],
            outputs="rf_metrics",
            name="evaluate_rf_node",
        ),
        Node(
            func=evaluate_model,
            inputs=["xgboost_model", "features_test", "params:model_names.xgboost"],
            outputs="xgb_metrics",
            name="evaluate_xgb_node",
        ),
        Node(
            func=evaluate_model,
            inputs=["lightgbm_model", "features_test", "params:model_names.lightgbm"],
            outputs="lgb_metrics",
            name="evaluate_lgb_node",
        ),
        Node(
            func=compare_models,
            inputs=["lr_metrics", "rf_metrics", "xgb_metrics", "lgb_metrics"],
            outputs="model_comparison",
            name="compare_models_node",
        ),
    ])
