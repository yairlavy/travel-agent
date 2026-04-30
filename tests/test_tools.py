"""
Integration tests for all SQL-backed tools.
These tests hit the real SQLite database — no mocks, no API calls.
"""

import pytest
from src.utils.db_init import create_travel_db

# Recreate DB at test start so tests are always idempotent
create_travel_db()

from src.tools.db_tools import (
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
    list_destinations,
    get_cheapest_hotel,
    get_cheapest_flight,
)
from src.tools.calc_tools import calculate_trip_cost


# ── fetch_flights ─────────────────────────────────────────────────────────────

def test_fetch_flights_found():
    result = fetch_flights.invoke({"origin": "TLV", "destination": "Paris"})
    assert "El Al" in result or "LY321" in result

def test_fetch_flights_case_insensitive():
    result = fetch_flights.invoke({"origin": "tlv", "destination": "paris"})
    assert "El Al" in result or "AF123" in result

def test_fetch_flights_not_found():
    result = fetch_flights.invoke({"origin": "TLV", "destination": "Sydney"})
    assert "No flights" in result


# ── fetch_hotels ──────────────────────────────────────────────────────────────

def test_fetch_hotels_found():
    result = fetch_hotels.invoke({"city": "Paris"})
    assert "Hotel de Ville" in result

def test_fetch_hotels_budget_filter_passes():
    result = fetch_hotels.invoke({"city": "Paris", "max_price": 90})
    assert "Ibis" in result

def test_fetch_hotels_budget_filter_excludes():
    result = fetch_hotels.invoke({"city": "Paris", "max_price": 50})
    assert "No hotels" in result

def test_fetch_hotels_unknown_city():
    result = fetch_hotels.invoke({"city": "Dubai"})
    assert "No hotels" in result


# ── fetch_activities ──────────────────────────────────────────────────────────

def test_fetch_activities_found():
    result = fetch_activities.invoke({"city": "Paris"})
    assert "Louvre" in result or "Eiffel" in result

def test_fetch_activities_unknown_city():
    result = fetch_activities.invoke({"city": "Dubai"})
    assert "No activities" in result


# ── get_visa_requirement ──────────────────────────────────────────────────────

def test_get_visa_requirement_known():
    result = get_visa_requirement.invoke(
        {"origin_country": "Israel", "destination_country": "France"}
    )
    assert "visa" in result.lower()

def test_get_visa_requirement_unknown_route():
    result = get_visa_requirement.invoke(
        {"origin_country": "Brazil", "destination_country": "Antarctica"}
    )
    assert "No visa information" in result


# ── list_destinations ─────────────────────────────────────────────────────────

def test_list_destinations_known_origin():
    result = list_destinations.invoke({"origin": "TLV"})
    for city in ("Paris", "London", "Berlin"):
        assert city in result

def test_list_destinations_unknown_origin():
    result = list_destinations.invoke({"origin": "XYZ"})
    assert "No destinations" in result


# ── get_cheapest_hotel ────────────────────────────────────────────────────────

def test_get_cheapest_hotel_berlin():
    result = get_cheapest_hotel.invoke({"city": "Berlin"})
    assert "40" in result or "Hostel" in result

def test_get_cheapest_hotel_unknown():
    result = get_cheapest_hotel.invoke({"city": "Dubai"})
    assert "No hotels" in result


# ── get_cheapest_flight ───────────────────────────────────────────────────────

def test_get_cheapest_flight_berlin():
    result = get_cheapest_flight.invoke({"origin": "TLV", "destination": "Berlin"})
    assert "110" in result or "Ryanair" in result

def test_get_cheapest_flight_not_found():
    result = get_cheapest_flight.invoke({"origin": "TLV", "destination": "Sydney"})
    assert "No flights" in result


# ── calculate_trip_cost ───────────────────────────────────────────────────────

def test_calculate_trip_cost_totals_correctly():
    # 350 (flight) + 150*5 (hotel) = 1100
    result = calculate_trip_cost.invoke(
        {"flight_price": 350.0, "hotel_price_per_night": 150.0, "duration_days": 5}
    )
    assert "1100" in result

def test_calculate_trip_cost_shows_breakdown():
    result = calculate_trip_cost.invoke(
        {"flight_price": 100.0, "hotel_price_per_night": 50.0, "duration_days": 3}
    )
    assert "flight" in result.lower()
    assert "hotel" in result.lower()

def test_calculate_trip_cost_bad_input():
    result = calculate_trip_cost.invoke(
        {"flight_price": "bad", "hotel_price_per_night": 50.0, "duration_days": 3}
    )
    assert "Error" in result or "error" in result
