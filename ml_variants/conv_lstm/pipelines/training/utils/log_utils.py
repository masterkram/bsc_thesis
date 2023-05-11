import logging
from datetime import datetime
from mlflow import MlflowClient


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

    log_message = {
        "debug": logging.debug,
        "info": logging.info,
        "warning": logging.warning,
        "error": logging.error,
        "critical": logging.critical,
    }

    timestring = datetime.utcnow().strftime("%d/%m/%YT%H:%M:%S.%fZ")

    log_message[level](f"[{timestring}] {message}")


def log_mlflow(run):
    client = MlflowClient()
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "model")]
    write_log("run_id: {}".format(run.info.run_id))
    write_log("artifacts: {}".format(artifacts))
    write_log("params: {}".format(run.data.params))
    write_log("metrics: {}".format(run.data.metrics))
    write_log("tags: {}".format(tags))
