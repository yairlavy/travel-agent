import json
import os
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_db.json")

def _load_db():
    with open(DB_PATH, "r") as f:
        return json.load(f)


@tool
def fetch_flights(origin: str, destination: str) -> str:
    """
    Fetch available flights between two cities from the local database.
    Returns a list of flights with airline name, flight number, price, and availability status.
    Use this tool when the user asks about flights, routes, or travel options between two cities.
    """
    db = _load_db()
    results = [
        f for f in db["flights"]
        if f["origin"].lower() == origin.lower()
        and f["destination"].lower() == destination.lower()
        and f.get("availability", "").lower() != "unavailable"
    ]
    if not results:
        return f"No flights found from {origin} to {destination}."
    return json.dumps(results, indent=2)


@tool
def fetch_hotels(city: str, max_price_per_night: float = None) -> str:
    """
    Fetch available hotels in a given city from the local database.
    Optionally filter by a maximum price per night (budget filter).
    Returns hotel name, stars, price per night, and amenities.
    Use this tool when the user asks about hotels, accommodation, or places to stay in a city.
    """
    db = _load_db()
    results = [h for h in db["hotels"] if h["city"].lower() == city.lower()]

    if max_price_per_night is not None:
        results = [h for h in results if h["price_per_night"] <= max_price_per_night]

    if not results:
        if max_price_per_night is not None:
            return f"No available hotels found in {city} within a budget of ${max_price_per_night} per night."
        return f"No available hotels found in {city}."
    return json.dumps(results, indent=2)


@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, num_nights: int) -> str:
    """
    Calculate the total cost of a trip including a 10% service fee.
    Takes the flight price, hotel price per night, and number of nights.
    Returns a breakdown of costs and the final total.
    Use this tool when the user asks about total trip cost, budget estimation, or price calculation.
    """
    hotel_total = hotel_price_per_night * num_nights
    subtotal = flight_price + hotel_total
    service_fee = subtotal * 0.10
    total = subtotal + service_fee

    breakdown = {
        "flight_price": f"${flight_price:.2f}",
        "hotel_total": f"${hotel_total:.2f} ({num_nights} nights x ${hotel_price_per_night:.2f})",
        "subtotal": f"${subtotal:.2f}",
        "service_fee_10%": f"${service_fee:.2f}",
        "total": f"${total:.2f}"
    }
    return json.dumps(breakdown, indent=2)


@tool
def list_destinations(origin: str) -> str:
    """
    List all available destinations from a given origin city.
    Use this tool when the user wants to know where they can fly from a specific city,
    or asks 'what destinations are available from X'.
    """
    db = _load_db()
    destinations = list({
        f["destination"] for f in db["flights"]
        if f["origin"].lower() == origin.lower()
    })
    if not destinations:
        return f"No available destinations found from {origin}."
    return f"Available destinations from {origin}: {', '.join(destinations)}"


@tool
def get_cheapest_hotel(city: str) -> str:
    """
    Find the cheapest hotel in a given city.
    Use this tool when the user asks for the most affordable or budget-friendly option in a city.
    """
    db = _load_db()
    hotels = [h for h in db["hotels"] if h["city"].lower() == city.lower()]
    if not hotels:
        return f"No hotels found in {city}."
    cheapest = min(hotels, key=lambda h: h["price_per_night"])
    return json.dumps(cheapest, indent=2)


@tool
def get_cheapest_flight(origin: str, destination: str) -> str:
    """
    Find the cheapest available flight between two cities.
    Use this tool when the user wants the best price or most affordable flight option.
    """
    db = _load_db()
    flights = [
        f for f in db["flights"]
        if f["origin"].lower() == origin.lower()
        and f["destination"].lower() == destination.lower()
    ]
    if not flights:
        return f"No flights found from {origin} to {destination}."
    cheapest = min(flights, key=lambda f: f["price"])
    return json.dumps(cheapest, indent=2)
