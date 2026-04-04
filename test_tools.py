import sys

class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

_log = open("test_results.txt", "w")
sys.stdout = Tee(_log)

from tools import (
    fetch_flights,
    fetch_hotels,
    calculate_trip_cost,
    list_destinations,
    get_cheapest_hotel,
    get_cheapest_flight,
    fetch_activities,
    get_visa_requirement,
)

PASS = "[PASS]"
FAIL = "[FAIL]"

def run_test(name, result, expect_contains):
    if expect_contains.lower() in result.lower():
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}")
        print(f"       Expected to contain: '{expect_contains}'")
    print(f"       Output: {result}")
    print()

print("\n=== Tool Tests ===\n")

# fetch_flights
run_test(
    "fetch_flights: London -> Paris",
    fetch_flights.invoke({"origin": "London", "destination": "Paris"}),
    "AF123"
)
run_test(
    "fetch_flights: no results",
    fetch_flights.invoke({"origin": "Paris", "destination": "Tokyo"}),
    "No flights found"
)

# fetch_hotels
run_test(
    "fetch_hotels: Paris hotels",
    fetch_hotels.invoke({"city": "Paris"}),
    "Hotel Ritz"
)
run_test(
    "fetch_hotels: Paris budget filter (max 100)",
    fetch_hotels.invoke({"city": "Paris", "max_price_per_night": 100}),
    "EcoStay"
)
run_test(
    "fetch_hotels: Paris budget filter too low",
    fetch_hotels.invoke({"city": "Paris", "max_price_per_night": 50}),
    "No available hotels found"
)
run_test(
    "fetch_hotels: unknown city",
    fetch_hotels.invoke({"city": "Dubai"}),
    "No available hotels found"
)

# calculate_trip_cost
# flight=120, hotel=85x3=255, subtotal=375, fee=37.5, total=412.5
run_test(
    "calculate_trip_cost: includes total",
    calculate_trip_cost.invoke({"flight_price": 120, "hotel_price_per_night": 85, "num_nights": 3}),
    "412.50"
)
run_test(
    "calculate_trip_cost: includes 10% service fee",
    calculate_trip_cost.invoke({"flight_price": 120, "hotel_price_per_night": 85, "num_nights": 3}),
    "37.50"
)

# list_destinations
run_test(
    "list_destinations: from London",
    list_destinations.invoke({"origin": "London"}),
    "Paris"
)
run_test(
    "list_destinations: no routes",
    list_destinations.invoke({"origin": "Dubai"}),
    "No available destinations"
)

# get_cheapest_hotel
run_test(
    "get_cheapest_hotel: Paris",
    get_cheapest_hotel.invoke({"city": "Paris"}),
    "EcoStay"
)
run_test(
    "get_cheapest_hotel: unknown city",
    get_cheapest_hotel.invoke({"city": "Sydney"}),
    "No hotels found"
)

# get_cheapest_flight
run_test(
    "get_cheapest_flight: London -> Paris",
    get_cheapest_flight.invoke({"origin": "London", "destination": "Paris"}),
    "AF123"
)
run_test(
    "get_cheapest_flight: no results",
    get_cheapest_flight.invoke({"origin": "Tokyo", "destination": "Paris"}),
    "No flights found"
)
# fetch_activities
run_test(
    "fetch_activities: Paris",
    fetch_activities.invoke({"city": "Paris"}),
    "Louvre"
)

run_test(
    "fetch_activities: unknown city",
    fetch_activities.invoke({"city": "Dubai"}),
    "No activities found"
)

# get_visa_requirement
run_test(
    "get_visa_requirement: Israel -> France",
    get_visa_requirement.invoke({"origin_country": "Israel", "destination_country": "France"}),
    "No visa required"
)

run_test(
    "get_visa_requirement: unknown route",
    get_visa_requirement.invoke({"origin_country": "Brazil", "destination_country": "Japan"}),
    "No visa information found"
)
print()
