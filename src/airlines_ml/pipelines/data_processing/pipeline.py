"""
Data processing pipeline for Airlines ML project.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import preprocess_airlines_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=preprocess_airlines_data,
            inputs="airlines_raw",
            outputs="airlines_processed",
            name="preprocess_airlines_data_node",
        ),
        Node(
            func=split_data,
            inputs="airlines_processed", 
            outputs=["airlines_train", "airlines_val", "airlines_test"],
            name="split_data_node",
        ),
    ])
