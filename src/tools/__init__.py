"""
Central tool registry — import ALL_TOOLS from here to bind to any agent.
"""

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
from src.tools.search_tools import web_search

ALL_TOOLS = [
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
    list_destinations,
    get_cheapest_hotel,
    get_cheapest_flight,
    calculate_trip_cost,
    web_search,
]

__all__ = ["ALL_TOOLS"] + [t.name for t in ALL_TOOLS]
