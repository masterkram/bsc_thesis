from sat_2_rad_pipeline import conv_lstm_pipeline
from steps.evaluators import evaluator
from steps.importers import importer_sat2rad
from steps.trainers import trainer
from steps.visualizers import visualize


def main():
    conv_lstm_pipeline(
        importer=importer_sat2rad(),
        trainer=trainer(),
        evaluator=evaluator(),
        visualizer=visualize(),
    ).run()


if __name__ == "__main__":
    main()
