from zenml.config import DockerSettings
from zenml.integrations.constants import PYTORCH, PYTORCH_L
from zenml.pipelines import pipeline

docker_settings = DockerSettings(
    required_integrations=[PYTORCH, PYTORCH_L], requirements=["torchvision"]
)


@pipeline(settings={"docker": docker_settings})
def mnist_pipeline(importer, trainer, evaluator):
    train_data_loader, test_data_loader, predict_data_loader = importer()
    model = trainer(train_data_loader)
    evaluator(test_data_loader, model=model)
