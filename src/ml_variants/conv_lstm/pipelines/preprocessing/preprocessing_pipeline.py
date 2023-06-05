from zenml import pipeline
import steps.radar as rad
import steps.satellite as sat
import steps.downloaders as d
import steps.loaders as l


@pipeline(enable_cache=False)
def preprocessing_pipeline():
    l.load_data.after(d.download_data)

    sat.get_satellite_stats.after(sat.reproject_satellite)
    sat.satellite_pixel_normalization.after(sat.get_satellite_stats)

    rad.normalize_radar_pixels.after(rad.convert_radar_to_numpy)
    rad.resize_radar_files.after(rad.normalize_radar_pixels)
    sat.visualize_satellite.after(sat.satellite_pixel_normalization)
    rad.visualize_radar.after(rad.resize_radar_files)

    d.download_data()

    satellite_images, radar_images = l.load_data()

    # satellite data
    sat.reproject_satellite(satellite_images)
    stats = sat.get_satellite_stats(satellite_images)
    sat.satellite_pixel_normalization(satellite_images, stats)

    # radar data
    rad.convert_radar_to_numpy(radar_images)
    rad.normalize_radar_pixels(radar_images)
    rad.resize_radar_files(radar_images)
    # visualize the data.
    sat.visualize_satellite(satellite_images)
    rad.visualize_radar(radar_images)


if __name__ == "__main__":
    preprocessing_pipeline()
