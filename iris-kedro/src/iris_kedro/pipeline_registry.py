from kedro.pipeline import Pipeline
from iris_kedro.pipelines.iris_training import pipeline as iris_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Register the project's pipelines.

    Returns:
        A mapping from pipeline names to Pipeline objects.
    """
    iris_training = iris_training_pipeline.create_pipeline()

    return {
        "__default__": iris_training,
        "iris_training": iris_training,
    }
