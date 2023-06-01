from zenml import pipeline
from steps.preprocessors import (
    load_data,
    download_data,
    preprocess_radar,
    preprocess_satellite,
    visualize_radar_data,
    visualize_satellite_data,
)

# from steps.statistics import get_statistics


@pipeline(enable_cache=False)
def preprocessing_pipeline():
    load_data.after(download_data)
    # download data and load in to the pipeline.

    download_data()

    satellite_images, radar_images = load_data()
    # sat_stats, rad_stats = get_statistics(satellite_images, radar_images)
    # preprocess two different type of data.
    preprocess_satellite(satellite_images)
    preprocess_radar(radar_images, {})
    # visualize the data.
    visualize_satellite_data(satellite_images)
    visualize_radar_data(radar_images)


if __name__ == "__main__":
    preprocessing_pipeline()
