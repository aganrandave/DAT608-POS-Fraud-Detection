import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))

from geo import haversine_km, haversine_km_or_none  # noqa: E402


def test_haversine_km_zero_distance():
    assert haversine_km(6.5244, 3.3792, 6.5244, 3.3792) == 0.0


def test_haversine_km_lagos_to_abuja_is_reasonable():
    lagos = (6.5244, 3.3792)
    abuja = (9.0765, 7.3986)
    distance = haversine_km(*lagos, *abuja)
    # Straight-line distance between Lagos and Abuja is roughly 480-540 km
    assert 400 < distance < 600


def test_haversine_km_is_symmetric():
    a = (6.5244, 3.3792)
    b = (9.0765, 7.3986)
    assert round(haversine_km(*a, *b), 4) == round(haversine_km(*b, *a), 4)


def test_haversine_km_or_none_returns_none_when_any_input_is_none():
    assert haversine_km_or_none(None, None, 6.5244, 3.3792) is None
    assert haversine_km_or_none(6.5244, 3.3792, None, None) is None
    assert haversine_km_or_none(None, 3.3792, 6.5244, 3.3792) is None


def test_haversine_km_or_none_matches_haversine_km_when_all_present():
    a = (6.5244, 3.3792)
    b = (9.0765, 7.3986)
    assert haversine_km_or_none(*a, *b) == haversine_km(*a, *b)
