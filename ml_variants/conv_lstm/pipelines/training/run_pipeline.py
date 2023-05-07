from conv_lstm_pipeline import conv_lstm_pipeline
from steps.evaluators import evaluator
from steps.importers import importer_mnist
from steps.trainers import trainer


def main():
    conv_lstm_pipeline(
        importer=importer_mnist(), trainer=trainer(), evaluator=evaluator()
    ).run()


if __name__ == "__main__":
    main()
