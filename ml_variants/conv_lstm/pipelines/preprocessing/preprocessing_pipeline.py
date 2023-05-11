from zenml.config import DockerSettings
from zenml.integrations.constants import PYTORCH, PYTORCH_L
from zenml.pipelines import pipeline


@pipeline()
def preprocessing_pipeline(preprocessor):
    preprocessor()
