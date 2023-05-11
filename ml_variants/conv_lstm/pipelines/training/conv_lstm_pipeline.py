from zenml.config import DockerSettings
from zenml.integrations.constants import PYTORCH, PYTORCH_L
from zenml.pipelines import pipeline

docker_settings = DockerSettings(
    required_integrations=[PYTORCH, PYTORCH_L], requirements=["torchvision"]
)


@pipeline(settings={"docker": docker_settings})
def conv_lstm_pipeline(importer, trainer, evaluator, visualizer):
    train_dataloader, test_dataloader, predict_dataloader = importer()
    model = trainer(train_dataloader)
    evaluator(test_dataloader=test_dataloader, model=model)
    visualizer(predict_dataloader)
