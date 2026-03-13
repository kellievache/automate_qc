#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:36:01 2026
@author: vachek
"""


import numpy as np
import pandas as pd
import math
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform as rio_transform
from rasterio.transform import array_bounds
from matplotlib import cm, colors
from PIL import Image
import folium
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from scipy.ndimage import label, uniform_filter, generate_binary_structure, distance_transform_edt, median_filter, gaussian_filter


"""
    Return the single nearest station to (lon, lat) from PRISM metadata.

    This executes a PostGIS nearest-neighbor query against:
        - meta.history (geometry column: the_geom)
        - meta.station (joined by history.station_id = station.id)

    It computes the distance in meters using geography and orders by the
    <-> KNN operator for fast nearest-neighbor retrieval, returning the
    single closest row (LIMIT 1).

    Parameters
    ----------
    lon : float
        Longitude (EPSG:4326), degrees East (negative for West).
    lat : float
        Latitude  (EPSG:4326), degrees North (negative for South).
    engine : sqlalchemy.engine.Engine, optional (keyword-only)
        Existing SQLAlchemy engine to use. Prefer passing this so you can
        reuse pooled connections efficiently across calls.
    conn_str : str, optional (keyword-only)
        Connection string to build a temporary engine if `engine` is not provided.
        Example: "postgresql://user:password@host:5432/dbname"
    raise_on_error : bool, default=False (keyword-only)
        If True, re-raise exceptions. If False, swallow and return an empty DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
            station_id, name, lon, lat, elev, stnid, network_id, distance_m
        It will contain exactly one row (nearest station) if found, or be empty if none found
        or if an error occurred and raise_on_error=False.
"""

def getNearestStation(lon,lat):
    sql=f"""
            WITH params AS (
              SELECT ST_SetSRID(ST_Point({lon}, {lat}), 4326)::geography AS g
            )
            SELECT
              t.station_id,
              t.name,
              t.lon,
              t.lat,
              t.elev,
              s.stnid,
              s.network_id,
              ST_Distance(t.the_geom::geography, p.g) AS distance_m
            FROM meta.history AS t
            JOIN meta.station AS s
              ON s.id = t.station_id
            CROSS JOIN params AS p
            ORDER BY t.the_geom::geography <-> p.g
            LIMIT 1;
            """
    
    engine2 = create_engine("postgresql://web:no_password@virtrat.nacse.org:5432/prism_virt")
    df=pd.DataFrame()
    with engine2.connect() as connection:
        try:
            df = pd.read_sql(sql, engine2) 
        except Exception as e:
            t='no station'
    engine2.dispose()
    return df

def read_hdr(hdr_path):
    """Parse ESRI .hdr file for georeferencing info."""
    info = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0].lower()
                value = parts[1]
                info[key] = value
    return info

def rowcol_from_latlon(hdr, lon, lat):
    """
    Convert lon/lat -> (row, col) using ESRI BIL header where
    ULXMAP/ULYMAP are the *center* of the upper-left pixel.

    Returns zero-based (row, col) suitable for NumPy indexing.
    """
    ULXMAP = float(hdr["ULXMAP"])
    ULYMAP = float(hdr["ULYMAP"])
    XDIM   = float(hdr["XDIM"])
    YDIM   = float(hdr["YDIM"])

    # Zero-based, pixel-center convention:
    r = (ULYMAP - lat) / YDIM
    c = (lon - ULXMAP) / XDIM

    # Map from continuous to discrete index:
    row = int(round(r))
    col = int(round(c))
    return row, col


def latlon_from_rowcol(hdr, row, col):
    """
    Convert zero-based (row, col) -> (lat, lon) at the *pixel center*
    using ESRI BIL header where ULXMAP/ULYMAP are pixel centers.
    """
    ULXMAP = float(hdr["ulxmap"])
    ULYMAP = float(hdr["ulymap"])
    XDIM   = float(hdr["xdim"])
    YDIM   = float(hdr["ydim"])

    lat = ULYMAP - row * YDIM
    lon = ULXMAP + col * XDIM
    return lat, lon


def pixel_corners_from_rowcol(hdr, row, col):
    """
    Get the *corner (edge)* coordinates of pixel (row, col), zero-based.
    Returns (south, west, north, east) in lat/lon.
    """
    ULXMAP = float(hdr["ulxmap"])
    ULYMAP = float(hdr["ulymap"])
    XDIM   = float(hdr["xdim"])
    YDIM   = float(hdr["ydim"])

    # Center
    lat_c = ULYMAP - row * YDIM
    lon_c = ULXMAP + col * XDIM

    half_x = XDIM / 2.0
    half_y = YDIM / 2.0

    south = lat_c - half_y
    north = lat_c + half_y
    west  = lon_c - half_x
    east  = lon_c + half_x
    return south, west, north, east

"""
find_clusters_window4

Goal:
    Find spatial clusters of "standout" anomalies—either warm (above a local upper
    threshold) or cold (below a local lower threshold)—and keep only clusters that
    are compact/round enough and large enough to matter.

Steps

Preconditions / inputs
    - data: 2-D array (H×W). If 3-D with a singleton axis, it is squeezed to 2-D.
    - Optional lat_grid/lon_grid or an hdr used with latlon_from_rowcol(...) to
      locate the extreme cell geographically.
    - Shape/quality parameters (window size, connectivity, geometric filters, etc.).

Local statistics & thresholds
    - Compute a local mean (Gaussian filter) and a local variance via uniform_filter
      applied to data**2, then var = E[x^2] - (E[x])^2.
    - Upper threshold = local_mean + k * local_std  (warm)
    - Lower threshold = local_mean - k * local_std  (cold)
    - A cell is "standout" if it is above the upper or below the lower threshold.

Connected components
    - Build a 4- or 8-connected structure and label contiguous standout pixels into
      candidate clusters.

Cluster filters (quality gates)
    - Min size: bounding box height/width >= min_rows/min_cols.
    - Fill ratio: area / bbox_area >= min_fill_ratio (avoids wispy shapes).
    - Axis ratio via PCA: sqrt(lambda_max / lambda_min) <= max_axis_ratio
      (reject highly elongated blobs).
    - Optional circularity (use_radius_ratio):
        r_max / r_eq >= min_radius_ratio,
        where r_max is the max inscribed radius from a distance transform, and
        r_eq = sqrt(area / pi) (equal-area disk).

Attributes for survivors
    - Determine type ("warm" vs "cold") using the sign of the mean anomaly relative
      to the local mean.
    - Find the extreme cell (max for warm / min for cold), and record row/col, value,
      and lat/lon (via latlon_from_rowcol in the current code).
    - Optionally enrich with nearest-station metadata using getNearestStation(...).

Relabel & return
    - Remap surviving cluster IDs to a compact label set (1..N).
    - Set background to -9999.
    - Return (labeled_array, cluster_info).

Parameters
    data : numpy.ndarray
        Input raster of anomalies (2-D). If 3-D with a singleton axis, it will be
        squeezed to 2-D.

    window_size : int, default=50
        Neighborhood size for local moment estimates (when using uniform filters).
        Controls the spatial scale of the local mean/variance.

    min_size : int | tuple[int, int], default=10
        Minimum cluster size in terms of bounding-box dimensions. If int, the same
        value is used for rows and cols. If tuple, interpreted as (min_rows, min_cols).

    k : float, default=1.05
        Local threshold multiplier for the standard deviation. Points are considered
        standout if they exceed local_mean ± k * local_std.

    connectivity : int, {4, 8}, default=4
        Pixel connectivity used for labeling connected components (rook=4, queen=8).

    max_axis_ratio : float, default=1.5
        Maximum allowed elongation of a cluster, computed from PCA eigenvalues as
        sqrt(lambda_max / lambda_min). Values close to 1.0 are more circular.

    min_fill_ratio : float, default=0.4
        Minimum area fraction relative to the cluster's bounding box (area / bbox_area).
        Filters out wispy or highly perforated shapes.

    use_radius_ratio : bool, default=False
        If True, apply an additional circularity test using the ratio of the maximum
        inscribed radius to the equivalent-disk radius.

    min_radius_ratio : float, default=0.6
        Threshold for the optional circularity metric. Keep clusters with
        r_max / r_eq >= min_radius_ratio, where r_eq = sqrt(area / pi).

    hdr : Any, optional
        Georeferencing header or metadata consumed by latlon_from_rowcol(hdr, r, c)
        to map array indices to geographic coordinates.

    lat_grid : numpy.ndarray | None, optional
        Optional 2-D latitude grid matching data.shape. If provided (with lon_grid),
        can be used to report the extreme cell’s latitude.

    lon_grid : numpy.ndarray | None, optional
        Optional 2-D longitude grid matching data.shape. If provided (with lat_grid),
        can be used to report the extreme cell’s longitude.

Returns
    labeled_array : numpy.ndarray
        2-D array of the same shape as `data` with contiguous cluster labels
        (1..N) for surviving clusters and -9999 for background.

    cluster_info : dict[int, dict]
        Mapping from cluster label -> attribute dict with fields:
            - "height": int, cluster bounding-box height (rows)
            - "width": int, cluster bounding-box width (cols)
            - "count": int, number of pixels in the cluster
            - "axis_ratio": float, sqrt(lambda_max / lambda_min) from PCA
            - "fill_ratio": float, area / bbox_area
            - "type": str, "warm" or "cold"
            - "extreme_value": float, max (warm) or min (cold) value in cluster
            - "extreme_row": int, row index of the extreme cell
            - "extreme_col": int, col index of the extreme cell
            - "extreme_lat": float, optional, latitude of extreme cell (if available)
            - "extreme_lon": float, optional, longitude of extreme cell (if available)
            - "radius_ratio": float, optional, r_max / r_eq if computed
"""

def find_clusters_window4(
    data,
    window_size=50,
    min_size=10,              # int or (min_rows, min_cols)
    k=1.05,                   # local threshold multiplier
    connectivity=4,           # 4 or 8 for 2D
    max_axis_ratio=1.5,       # <= 1.0 is a perfect circle; allow up to 1.5 elongation
    min_fill_ratio=0.4,       # area / (bbox_area) must exceed this
    use_radius_ratio=False,   # optional stronger circularity check
    min_radius_ratio=0.6,     # r_max / r_eq (inscribed vs equivalent radius)
    hdr=None,
    lat_grid=None,            # optional 2D latitude grid matching data shape
    lon_grid=None             # optional 2D longitude grid matching data shape
):

    # --- Normalize input to 2D ---
    data = np.squeeze(data)  # handles (H, W, 1) -> (H, W)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze; got {data.ndim}D.")

    # Validate lat/lon grids if provided
    if (lat_grid is not None) or (lon_grid is not None):
        if (lat_grid is None) or (lon_grid is None):
            raise ValueError("Provide both lat_grid and lon_grid, or neither.")
        if lat_grid.shape != data.shape or lon_grid.shape != data.shape:
            raise ValueError(f"lat_grid/lon_grid must match data shape {data.shape}.")

    # --- Normalize min_size ---
    if isinstance(min_size, int):
        min_rows = min_cols = min_size
    else:
        min_rows, min_cols = min_size

    # --- Local stats & thresholds ---
    #local_mean = uniform_filter(data, size=window_size)
   # local_mean = median_filter(data, size=window_size)
    local_mean = gaussian_filter(data, sigma=5)
    local_mean_sq = uniform_filter(data**2, size=window_size)
    local_var = local_mean_sq - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))  # numerical guard
    
    
    plt.figure(figsize=(8, 6))
    local_mean = np.ma.masked_equal(local_mean, -9999)
    plt.imshow(
        local_mean,
        cmap='viridis',
        origin='upper',
     #   extent=(lon_min, lon_max, lat_min, lat_max),
        interpolation='nearest'
    )
    plt.colorbar(label='Value')
    plt.title('local mean')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()
    
    
    

    upper_threshold = local_mean + k * local_std  # warm clusters
    lower_threshold = local_mean - k * local_std  # cold clusters

    # Boolean mask for standout cells (warm OR cold)
    standout_mask = (data > upper_threshold) | (data < lower_threshold)

    # --- Connectivity structure ---
    level = 1 if connectivity == 4 else 2 if connectivity == 8 else None
    if level is None:
        raise ValueError("connectivity must be 4 or 8 for 2D inputs.")
    structure = generate_binary_structure(rank=data.ndim, connectivity=level)

    # --- Label connected components ---
    labeled_array, num_features = label(standout_mask.astype(bool), structure=structure)

    cluster_info = {}
    for cluster_id in range(1, num_features + 1):
        rows, cols = np.where(labeled_array == cluster_id)
        if rows.size == 0:
            continue

        # --- Bounding box & min dimension check ---
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        height = max_r - min_r + 1
        width  = max_c - min_c + 1
        area   = rows.size
        bbox_area = height * width

        if height < min_rows or width < min_cols:
            labeled_array[labeled_array == cluster_id] = 0
            continue

        # --- Fill ratio ---
        fill_ratio = area / float(bbox_area)
        if fill_ratio < min_fill_ratio:
            labeled_array[labeled_array == cluster_id] = 0
            continue

        # --- Axis ratio via PCA ---
        r_center = rows.mean()
        c_center = cols.mean()
        coords = np.column_stack((rows - r_center, cols - c_center))
        cov = np.cov(coords, rowvar=False)
        if not np.isfinite(cov).all():
            labeled_array[labeled_array == cluster_id] = 0
            continue
        evals, _ = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-12)
        axis_ratio = np.sqrt(evals.max() / evals.min())
        if axis_ratio > max_axis_ratio:
            labeled_array[labeled_array == cluster_id] = 0
            continue

        # --- Optional: radius ratio ---
        radius_ratio = None
        if use_radius_ratio:
            mask_cluster = (labeled_array == cluster_id)
            r_max = distance_transform_edt(mask_cluster).max()
            r_eq  = np.sqrt(area / np.pi)
            if r_eq > 0:
                radius_ratio = r_max / r_eq
                if radius_ratio < min_radius_ratio:
                    labeled_array[labeled_array == cluster_id] = 0
                    continue

        # --- Determine cluster type (warm or cold) ---
        cluster_values = data[rows, cols]
        mean_anomaly = cluster_values.mean() - local_mean[rows, cols].mean()
        cluster_type = "warm" if mean_anomaly > 0 else "cold"

        # --- Find extreme cell within the cluster ---
        extreme_idx = np.argmax(cluster_values) if cluster_type == "warm" else np.argmin(cluster_values)
        extreme_row = int(rows[extreme_idx])
        extreme_col = int(cols[extreme_idx])
        extreme_value = float(data[extreme_row, extreme_col])

        # Map to lat/lon if grids are provided
     #   extreme_lat = float(lat_grid[extreme_row, extreme_col]) if lat_grid is not None else None
     #   extreme_lon = float(lon_grid[extreme_row, extreme_col]) if lon_grid is not None else None
        
        extreme_lat,extreme_lon = latlon_from_rowcol(hdr, extreme_row, extreme_col)

        # --- Keep cluster ---
        info = {
            "height": height,
            "width": width,
            "count": area,
            "axis_ratio": float(axis_ratio),
            "fill_ratio": float(fill_ratio),
            "type": cluster_type,
            "extreme_value": extreme_value,
            "extreme_row": extreme_row,
            "extreme_col": extreme_col,
        }
        if extreme_lat is not None and extreme_lon is not None:
            info["extreme_lat"] = extreme_lat
            info["extreme_lon"] = extreme_lon
        if radius_ratio is not None:
            info["radius_ratio"] = float(radius_ratio)
            

        extreme_lat,extreme_lon=latlon_from_rowcol(hdr, extreme_row, extreme_col)    
        station_data=getNearestStation(extreme_lon,extreme_lat)
        info['station_id']=station_data['station_id'][0]
        info['name']=station_data['name'][0]
        info['network_id']=station_data['network_id'][0]
        info['distance_m']=station_data['distance_m'][0]
        cluster_info[cluster_id] = info

    # --- Re-label surviving clusters contiguously ---
    if cluster_info:
        surviving = sorted(cluster_info.keys())
        remap = {old: new for new, old in enumerate(surviving, start=1)}
        for old, new in remap.items():
            labeled_array[labeled_array == old] = new
        cluster_info = {remap[old]: info for old, info in cluster_info.items()}
    labeled_array[labeled_array == 0 ] = -9999
    return labeled_array, cluster_info

def developMap_fromArray(
    data,
    transform,
    crs,
    nodata=-9999,                     # default as requested
    png_path_cluster="output/overlay_3857.png",
    html_path="output/cluster_map_overlay_3857.html",
    class_colors=None,                # {class_value: (R,G,B)}, 0-255; nodata is ignored
    fallback_cmap="tab20",            # used for any classes not in class_colors
    zoom_start=4
):
    """
    Reproject a *classified* raster array to EPSG:3857 (nearest-neighbor),
    render by class while masking NoData (-9999 by default) so it is not
    displayed at all (transparent in the PNG), and create a Folium overlay.
    """

    # --- Normalize input to one band ---
    if data.ndim == 3:
        band = data[..., 0]
    elif data.ndim == 2:
        band = data
    else:
        raise ValueError("`data` must be 2D (rows, cols) or 3D (rows, cols, bands).")

    band = np.asarray(band)

    # --- 1) Reproject to Web Mercator (EPSG:3857) using NEAREST ---
    dst_crs = "EPSG:3857"
    rows, cols = band.shape
    left, bottom, right, top = array_bounds(rows, cols, transform)

    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs, dst_crs, cols, rows, left, bottom, right, top
    )

    # Use NaN as destination NoData to make masking unambiguous
    data_3857 = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    reproject(
        source=band.astype(np.float32, copy=False),
        destination=data_3857,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=nodata,
        dst_nodata=np.nan,
    )

    # --- 2) Compute geographic bounds for Folium overlay ---
    left_m, bottom_m, right_m, top_m = array_bounds(data_3857.shape[0], data_3857.shape[1], dst_transform)
    (sw_lon,), (sw_lat,) = rio_transform(dst_crs, "EPSG:4326", [left_m],  [bottom_m])
    (ne_lon,), (ne_lat,) = rio_transform(dst_crs, "EPSG:4326", [right_m], [top_m])
    folium_bounds = [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # --- 3) Strict masking of NoData and class color rendering ---
    arr = data_3857  # float32 with NaN representing NoData
    mask = ~np.isfinite(arr)          # True where NoData

    # Build the output RGBA; start fully transparent everywhere
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    # Work only on valid pixels (non-NoData)
    valid = ~mask
    if np.any(valid):
        # Recover class codes (integers) after reprojection
        classes_present = np.unique(np.round(arr[valid]).astype(np.int64))

        # Prepare color map
        if class_colors is None:
            # Assign stable colors from a qualitative colormap
            mpl_cmap = cm.get_cmap(fallback_cmap, max(len(classes_present), 1))
            color_map = {
                int(cls): tuple(int(c * 255) for c in mpl_cmap(i % mpl_cmap.N)[:3])
                for i, cls in enumerate(classes_present)
            }
        else:
            # Normalize keys to int and ensure RGB tuples
            color_map = {int(k): tuple(v) for k, v in class_colors.items()}
            # Assign fallback colors to any classes not covered
            missing = [int(c) for c in classes_present if int(c) not in color_map]
            if missing:
                mpl_cmap = cm.get_cmap(fallback_cmap, max(len(missing), 1))
                for i, cls in enumerate(missing):
                    color_map[cls] = tuple(int(c * 255) for c in mpl_cmap(i % mpl_cmap.N)[:3])

        # Paint valid pixels: look up per-pixel class color
        cls_codes = np.round(arr[valid]).astype(np.int64)
        rgb = np.zeros((cls_codes.size, 3), dtype=np.uint8)
        for idx, cls in enumerate(cls_codes):
            rgb[idx] = color_map[int(cls)]
        rgba[valid, :3] = rgb
        rgba[valid, 3] = 255  # opaque for valid pixels

        # Note: nodata pixels remain alpha=0 (not shown at all)
        classes_list = classes_present.tolist()
    else:
        # No valid pixels; keep a fully transparent image
        color_map = {}
        classes_list = []

    # Write PNG (NoData is fully transparent)
    Image.fromarray(rgba, mode="RGBA").save(png_path_cluster)

    # --- 4) Folium overlay ---
    center = [(sw_lat + ne_lat) / 2, (sw_lon + ne_lon) / 2]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        image=png_path_cluster,
        bounds=folium_bounds,
        opacity=1.0,                 # full opacity; transparency is per-pixel via alpha
        interactive=True,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(html_path)

    # Build return info (NoData not included)
    legend = {int(k): (*v, 255) for k, v in color_map.items()}  # RGBA with full opacity

    print(f"Saved {html_path} and {png_path}")
    return {
        "dst_crs": dst_crs,
        "dst_transform": dst_transform,
        "classes": classes_list,     # NoData not included
        "color_map": legend          # No NoData entry here either
    }

def developMap(bil_path,
               png_path,
               html_path="cluster_map_overlay_3857.html",
               cmap_name="terrain",
               zoom_start=4):
    """
    Read a raster (BIL/ENVI or similar readable by rasterio), reproject to Web Mercator,
    colorize to a PNG with transparency over NoData, and create a Folium overlay.

    Parameters
    ----------
    bil_path : str
        Path to input raster.
    png_path : str
        Output PNG path.
    html_path : str
        Output Folium HTML path.
    cmap_name : str
        Matplotlib colormap name (e.g., 'terrain', 'viridis').
    zoom_start : int
        Initial Folium zoom level.

    Returns
    -------
    dst_crs, dst_transform : (str, Affine)
        CRS string and Affine transform of the EPSG:3857 image.
    """

    # --- 1) Read and warp to Web Mercator (EPSG:3857) ---
    with rasterio.open(bil_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        nodata = src.nodata  # may be None
        dtype = src.dtypes[0]

        dst_crs = "EPSG:3857"
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Use float32 + NaN as destination NoData for easy masking
        data_3857 = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=data_3857,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,     # carry forward source NoData
            dst_nodata=np.nan      # store as NaN so we can mask cleanly
        )

    # --- 2) Bounds in 3857 meters, convert corners to lat/lon for Folium ---
    left, bottom, right, top = array_bounds(
        data_3857.shape[0], data_3857.shape[1], dst_transform
    )

    # Convert SW and NE corners from EPSG:3857 → EPSG:4326
    (sw_lon,), (sw_lat,) = rio_transform(dst_crs, "EPSG:4326", [left],  [bottom])
    (ne_lon,), (ne_lat,) = rio_transform(dst_crs, "EPSG:4326", [right], [top])
    folium_bounds = [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # --- 3) Colorize to PNG (with transparency over NoData) ---
    arr = data_3857.astype("float32", copy=False)

    # Build mask: NaNs always masked; also mask any values equal to original nodata (defensive)
    mask = ~np.isfinite(arr)
    if nodata is not None:
        if np.issubdtype(type(nodata), np.floating):
            mask |= np.isclose(arr, nodata, rtol=0.0, atol=1e-7, equal_nan=False)
        else:
            mask |= (arr == nodata)

    # Robust contrast stretch on valid pixels only
    valid = ~mask
    if np.any(valid):
        vmin = float(np.nanpercentile(arr[valid], 2))
        vmax = float(np.nanpercentile(arr[valid], 98))
        if vmin == vmax:
            vmax = vmin + 1e-6
    else:
        # All masked: choose harmless defaults to avoid divide-by-zero
        vmin, vmax = 0.0, 1.0

    normed = (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin)
    rgba = cm.get_cmap(cmap_name)(normed, bytes=True)
    rgba = rgba.copy()  # ensure writeable
    rgba[mask] = (0, 0, 0, 0)  # fully transparent where NoData

    Image.fromarray(rgba, mode="RGBA").save(png_path)

    # --- 4) Folium overlay using lat/lon bounds of the Mercator image ---
    center = [(sw_lat + ne_lat) / 2, (sw_lon + ne_lon) / 2]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        image=png_path,
        bounds=folium_bounds,     # SW/NE of the Mercator-projected image
        opacity=0.7,
        interactive=True
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(html_path)
    print(f"Saved {html_path} and {png_path}")

    return src_crs, src_transform, center, folium_bounds

bil_path = "/nfs/pancake/prism_current/us/an/ehdr/800m/tmax/daily/2026/prism_tmax_us_30s_20260115.bil"          # <- your .bil
bil_path_out = "output/clusters.bil"  
png_path = "output/elevation_overlay.png" 
png_path_cluster = "output/overlay_3857.png"

nor_path="/nfs/pancake/prism_current/us/an/ehdr/800m/tmax/daily/normals/prism_tmax_us_30s_20200115_avg_30y.bil"    
hdr_path="/nfs/pancake/prism_current/us/an/ehdr/800m/tmax/daily/2026/prism_tmax_us_30s_20260115.hdr"

html_path="output/cluster_map.html"

hdr=read_hdr(hdr_path)

nrows=3105
ncols=7025
nbands=1
total_elems = nrows * ncols * nbands
data = np.fromfile(bil_path, dtype=np.float32, count=total_elems)
data = data.reshape((nrows, ncols, nbands))
norm = np.fromfile(nor_path, dtype=np.float32, count=total_elems)
norm = norm.reshape((nrows, ncols, nbands))
anom = data - norm

labels, clusters = find_clusters_window4(
    anom,
    window_size=250,
    min_size=(25, 25),
    k=4,
    connectivity=8,
    max_axis_ratio=2.,
    min_fill_ratio=0.5,
    use_radius_ratio=True,
    hdr=hdr,
    lat_grid=None,
    lon_grid=None
)

# develop Map reads and projects the prism grid.  The projection is needed because we want to overlay the clusters on top of the original data, 
# and the original data is in a geographic coordinate system but the folium map is in something else.  
# currently we are returning info from that process, and using that info in developMap_fromArray.
# thats clunky and they should be integrated, but this is a first pass. 
crs, transform, center , folium_bounds = developMap(bil_path,png_path)

#Develop a map overlay of the clusters using the original BIL for georeferencing and the labels for rendering. 
result = developMap_fromArray(
    data=labels,                 # can pass the 3D array; function will pick band 1
    transform=transform,
    crs=crs,
    nodata=-9999,
    png_path_cluster="output/overlay_3857.png",
    html_path="output/overlay_map.html",
   # cmap_name="terrain",

    class_colors=None,        # or None to auto-assign
    fallback_cmap="tab20",
    
    zoom_start=6
)

m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
folium.raster_layers.ImageOverlay(
    name="Temperature",
    image=png_path,
    bounds=folium_bounds,     # SW/NE of the Mercator-projected image
    opacity=0.7,
    interactive=True
).add_to(m)
#folium.LayerControl().add_to(m)


# Second overlay (same image, different name)
folium.raster_layers.ImageOverlay(
    name="Clusters",
    image=png_path_cluster,                   # same image as requested
    bounds=folium_bounds,
    opacity=0.7,
    interactive=True,
    zindex=2                          # draw above Overlay A
).add_to(m)


# --- NEW: add a FeatureGroup for Stations using lat/lon directly ---
stations_fg = folium.FeatureGroup(name="Stations", show=True)

for cid, info in clusters.items():
    lat = info.get('extreme_lat')
    lon = info.get('extreme_lon')
    name = info.get('name', f'Cluster {cid}')
    if lat in (None, 'N/A') or lon in (None, 'N/A'):
        continue

    folium.CircleMarker(
        location=(lat, lon),
        radius=6,
        color='black', weight=1,
        fill=True, fill_color='white', fill_opacity=1.0,
        tooltip=name,  # hover
        popup=folium.Popup(name, max_width=250)
    ).add_to(stations_fg)

stations_fg.add_to(m)


folium.LayerControl(collapsed=False).add_to(m)
m.save(html_path)
print(f"Saved {html_path} and {png_path}")
#print("Classes rendered:", result["classes"])

print(f"Found {len(clusters)} clusters:")
for cid, info in clusters.items():
    print(
        f"Cluster {cid}: {info['count']} cells, type={info['type']}, "
        f"extreme_value={info['extreme_value']:.2f}, "
        f"lat={info.get('extreme_lat', 'N/A')}, lon={info.get('extreme_lon', 'N/A')}, "
        f"station_id={info['station_id']}, "
        f"name={info['name']}, "
        f"network_id={info['network_id']}, "
        f"distance_m={info['distance_m']:.2f}, "
        
    )
    
   
    
subset2d=data   
    
# Visualize clusters overlay
subset2d = np.ma.masked_equal(subset2d, -9999)
plt.figure(figsize=(8, 6))
plt.imshow(subset2d, cmap='viridis', origin='upper')
plt.imshow(labels, cmap='jet', alpha=0.4, origin='upper')  # Overlay clusters
plt.colorbar(label='Temperature')
plt.title('Clusters of Warm/Cold Anomalies')
plt.tight_layout()
plt.show()


# --- Your existing plotting code ---
plt.figure(figsize=(10, 7))  # a little taller to make space for the table
plt.imshow(subset2d, cmap='viridis', origin='upper')
plt.imshow(labels, cmap='jet', alpha=0.4, origin='upper')  # Overlay clusters
plt.colorbar(label='Temperature')
plt.title('Clusters of Warm/Cold Anomalies')


# Optionally: plot the station points and annotate (from prior step)
xy_points = []
names = []
for cid, info in clusters.items():
    lat = info.get('extreme_lat')
    lon = info.get('extreme_lon')
    name = info.get('name', f'Cluster {cid}')
    if lat in (None, 'N/A') or lon in (None, 'N/A'):
        continue

if xy_points:
    xs = [pt[0] for pt in xy_points]
    ys = [pt[1] for pt in xy_points]
    plt.scatter(xs, ys, s=60, c='white', edgecolors='black', zorder=3, label='Stations')
    for (x, y), name in zip(xy_points, names):
        plt.annotate(
            name, (x, y),
            textcoords="offset points", xytext=(5, -5),
            ha='left', va='top', fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.6)
        )
plt.legend(loc='upper right')

# --- Build a table from the clusters dict and add it below the figure ---

# Choose columns to show (adjust as needed)
columns = [
     "name", "station_id", "network_id",
    "type", 
    "distance_m"
]

# Extract rows with safe defaults and formatting
rows = []

for cid, info in clusters.items():
    rows.append({
      
        "name": info.get("name", "")[:10],
        "station_id": info.get("station_id", ""),
        "network_id": info.get("network_id", ""),
        "type": info.get("type", ""),
        "distance_m": (
            f"{info['distance_m']:.1f}" if isinstance(info.get("distance_m"), (int, float)) else info.get("distance_m", "")
        ),
    })

# Create a DataFrame for convenience (optional but helps with ordering)
df = pd.DataFrame(rows, columns=columns)

# Limit the number of rows if you have many clusters (optional)
max_rows = 12
if len(df) > max_rows:
    df = df.iloc[:max_rows].copy()

# Render the table below the axes
# 1) Shrink the main axes to make vertical space for the table
ax = plt.gca()
plt.subplots_adjust(bottom=0.25)  # increase if the table is tall


# 2) Create the table
table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='lower center',         # place at bottom center of the axes
    cellLoc='center',
    colLoc='center',
    bbox=[0.0, -0.30, 1.0, 0.22]  # [left, bottom, width, height]; reduce height to shrink cells
)

# Styling tweaks
table.auto_set_font_size(False)
table.set_fontsize(7)           # smaller font to fit more text

# Shrink cell size uniformly: x-scale (width), y-scale (height)
# Values < 1.0 make cells smaller. Adjust as needed.
table.scale(0.9, 0.8)

# Optional: ensure columns aren’t too wide if df has very short content
# (Requires Matplotlib >= 3.7 for `auto_set_column_width`.)
# try:
#     table.auto_set_column_width(col=list(range(len(df.columns))))
# except Exception:
#     pass

# Bold header row and give a light background
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E6E6E6')  # light gray for header
    else:
        # zebra striping (optional)
        if row % 2 == 0:
            cell.set_facecolor('#F9F9F9')

# Make more room at the bottom if needed
plt.subplots_adjust(bottom=0.28)  # increase bottom margin so table isn't clipped

# Final layout
plt.tight_layout()
plt.show()

plt.savefig("output/Map_Table_QC.png", dpi=300, bbox_inches="tight")

