from zenml.config import DockerSettings
from zenml.integrations.constants import PYTORCH
from zenml.pipelines import pipeline

docker_settings = DockerSettings(
    required_integrations=[PYTORCH],
    requirements=["torchvision"]
)


@pipeline(settings={"docker": docker_settings})
def mnist_pipeline(importer, trainer, evaluator):
    train_dataloader, test_dataloader = importer()
    model = trainer(train_dataloader)
    evaluator(test_dataloader=test_dataloader, model=model)
