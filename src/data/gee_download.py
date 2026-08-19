"""
Google Earth Engine data acquisition utilities for the Delhi NCR
Urban Change Intelligence project.

This module provides functions to define the study area, query
Sentinel-2 Level-2A imagery, inspect basic metadata, and export
composite images to Google Drive.

Scene-level cloud filtering (via the CLOUDY_PIXEL_PERCENTAGE metadata
property) is applied here. Pixel-level cloud masking is a Phase 2
(preprocessing) task and is NOT performed by this module.

Intended to be run inside a Google Colab environment with Earth Engine
already authenticated and initialized (see notebooks/00_colab_setup.ipynb).
"""

import ee


def get_delhi_ncr_roi():
    """
    Return an approximate rectangular bounding box covering the Delhi
    NCR study area: Delhi, Gurugram, Noida, Ghaziabad, Faridabad.

    NOTE: This is a rectangular bounding box, not an administrative
    boundary. It is intentionally simple for Phase 1 data acquisition.
    Refining to precise administrative boundaries (if needed for area
    or region-level statistics) is deferred to Phase 9 (spatial
    intelligence), where it can be swapped in using geopandas + a
    shapefile without changing this function's interface.

    Returns:
        ee.Geometry.Rectangle: [xmin, ymin, xmax, ymax] in EPSG:4326.
    """
    xmin, ymin, xmax, ymax = 76.75, 28.30, 77.60, 28.95
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])


# Bands used throughout the project:
# B2  = Blue   (10m)
# B3  = Green  (10m)
# B4  = Red    (10m)
# B8  = NIR    (10m)
# B11 = SWIR1  (20m native, resampled to 10m on export)
# B12 = SWIR2  (20m native, resampled to 10m on export)
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def get_sentinel2_collection(start_date, end_date, roi, cloud_threshold=20):
    """
    Query Sentinel-2 Level-2A (surface reflectance) imagery for a given
    date range and region, filtered by scene-level cloud percentage.

    Args:
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        roi (ee.Geometry): region of interest
        cloud_threshold (float): max CLOUDY_PIXEL_PERCENTAGE (scene-level)

    Returns:
        ee.ImageCollection: filtered collection, selected bands only.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .select(BANDS)
    )
    return collection


def get_collection_metadata(collection):
    """
    Compute basic, honest metadata about a filtered collection: how many
    images survived the cloud filter, their individual scene-level cloud
    percentages, and acquisition dates. This feeds the Phase 1 data
    validation report — every number here comes directly from GEE, none
    are invented or estimated.

    Args:
        collection (ee.ImageCollection)

    Returns:
        dict with 'image_count', 'cloud_percentages', 'acquisition_dates'
    """
    image_count = collection.size().getInfo()
    cloud_percentages = collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    raw_dates = collection.aggregate_array("system:time_start").getInfo()
    acquisition_dates = [ee.Date(d).format("YYYY-MM-dd").getInfo() for d in raw_dates]

    return {
        "image_count": image_count,
        "cloud_percentages": cloud_percentages,
        "acquisition_dates": acquisition_dates,
    }


def get_median_composite(collection):
    """
    Create a cloud-reduced median composite from a filtered collection.
    This is a simple, defensible way to get one representative image per
    time period at this stage. Pixel-level cloud masking (Phase 2) will
    improve on this later — this is Phase 1's "good enough to inspect
    and export" composite, not a final preprocessed product.

    Args:
        collection (ee.ImageCollection)

    Returns:
        ee.Image: median composite, band names preserved.
    """
    return collection.median()


def export_to_drive(image, description, roi, folder="delhi_ncr_change_detection", scale=10):
    """
    Start an Earth Engine export task sending an image to Google Drive
    as a GeoTIFF. This is asynchronous — check the 'Tasks' tab at
    https://code.earthengine.google.com, or Colab's Runtime > Manage
    Sessions area, to monitor progress. Exports typically take several
    minutes depending on ROI size and scale.

    Args:
        image (ee.Image): image to export (e.g. a median composite)
        description (str): task name AND output filename (no extension)
        roi (ee.Geometry): export region
        folder (str): destination folder name inside Google Drive
        scale (int): output resolution in meters (10 = native Sentinel-2 RGB/NIR)

    Returns:
        ee.batch.Task: the started export task (already .start()-ed)
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=roi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    return task