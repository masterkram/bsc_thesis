from preprocessing_pipeline import preprocessing_pipeline
import steps.preprocessors as p


def main():
    preprocessing_pipeline(
        p.download_data(),
        p.load_data(),
        p.preprocess_satellite(),
        p.preprocess_radar(),
        p.visualize_satellite_data(),
        p.visualize_radar_data(),
    ).run()


if __name__ == "__main__":
    main()
