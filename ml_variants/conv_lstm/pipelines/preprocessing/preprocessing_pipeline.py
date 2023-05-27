from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def preprocessing_pipeline(
    download_step,
    load_step,
    statistics_step,
    pre_sat_step,
    pre_radar_step,
    viz_radar_step,
    viz_sat_step,
):
    load_step.after(download_step)
    # download data and load in to the pipeline.

    download_step()

    satellite_images, radar_images = load_step()
    sat_stats, rad_stats = statistics_step(satellite_images, radar_images)
    # preprocess two different type of data.
    pre_sat_step(satellite_images)
    pre_radar_step(radar_images, rad_stats)
    # visualize the data.
    viz_sat_step(satellite_images)
    viz_radar_step(radar_images)
