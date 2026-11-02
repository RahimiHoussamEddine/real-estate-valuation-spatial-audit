"""Create target-free geographic clusters for spatial cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from real_estate_audit.constants import (
    COORDINATE_COLUMNS,
    N_SPATIAL_CLUSTERS,
    RANDOM_STATE,
)


def local_metric_coordinates(frame: pd.DataFrame) -> np.ndarray:
    latitude = frame[COORDINATE_COLUMNS[0]].to_numpy(dtype=float)
    longitude = frame[COORDINATE_COLUMNS[1]].to_numpy(dtype=float)
    mean_latitude = float(np.mean(latitude))
    east_meters = (
        (longitude - float(np.mean(longitude)))
        * 111_320.0
        * np.cos(np.deg2rad(mean_latitude))
    )
    north_meters = (latitude - mean_latitude) * 110_574.0
    return np.column_stack((east_meters, north_meters))


def assign_spatial_clusters(frame: pd.DataFrame) -> np.ndarray:
    coordinates = local_metric_coordinates(frame)
    model = KMeans(
        n_clusters=N_SPATIAL_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    return model.fit_predict(coordinates)
