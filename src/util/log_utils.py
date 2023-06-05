import logging
from datetime import datetime
from mlflow import MlflowClient
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
from rich import print


def write_log(message: str, level: str = "info") -> None:
    """
    Write a message to logs.

    Attributes
    ----------
    message: str
        Message to write to logs.
    level: str
        Log level. Supported: ['debug', 'info', 'warning', 'error', 'critical']

    Returns
    -------
    Void.
    """
    if level not in ["debug", "info", "warning", "error", "critical"]:
        raise ValueError(f"Log level {level} is unsupported.")

    # timestring = datetime.utcnow().strftime("%d/%m/%YT%H:%M:%S.%fZ")

    log = logging.getLogger("rich")

    # log.info(message, extra={"markup": True})
    print(message)

    # log_message[level](f"[{timestring}] {message}")


def log_mlflow(run):
    client = MlflowClient()
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "model")]
    write_log("run_id: {}".format(run.info.run_id))
    write_log("artifacts: {}".format(artifacts))
    write_log("params: {}".format(run.data.params))
    write_log("metrics: {}".format(run.data.metrics))
    write_log("tags: {}".format(tags))
