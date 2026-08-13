"""Pure-Python geo math shared by feature_windows.py.

Kept dependency-free (no pyspark import) so it can be unit tested without
a Spark runtime.
"""
from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def haversine_km_or_none(lat1, lon1, lat2, lon2):
    """None-tolerant wrapper for callers (feature_windows.with_geo_jump's
    Spark UDF) that may receive None for a "no previous row" case - PySpark
    evaluates a Python UDF's arguments for every row before F.when/otherwise
    picks a branch, so haversine_km itself would crash on the row that's
    ultimately discarded rather than never being called."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    return haversine_km(lat1, lon1, lat2, lon2)
