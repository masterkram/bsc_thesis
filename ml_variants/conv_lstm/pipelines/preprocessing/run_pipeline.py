from preprocessing_pipeline import preprocessing_pipeline
from steps.preprocessors import preprocessor


def main():
    preprocessing_pipeline(
        preprocessor=preprocessor(),
    ).run()


if __name__ == "__main__":
    main()
