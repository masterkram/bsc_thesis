from zenml.config import DockerSettings
from zenml.integrations.constants import PYTORCH, PYTORCH_L
from zenml import pipeline
import steps.evaluators as ev
import steps.importers as im
import steps.trainers as tr
import steps.visualizers as vi

docker_settings = DockerSettings(
    required_integrations=[PYTORCH, PYTORCH_L], requirements=["torchvision"]
)


@pipeline(settings={"docker": docker_settings})
def movingmnist_pipeline():
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        predict_dataloader,
    ) = im.importer_sat2rad()
    model = tr.trainer(train_dataloader, val_dataloader)
    ev.evaluator(test_dataloader=test_dataloader, model=model)
    vi.visualize(predict_dataloader, model)


if __name__ == "__main__":
    movingmnist_pipeline()
