#!/usr/bin/env python3
"""This modlue Uses swap api to return a list fo ships that can carry a
given number of passengers"""
import requests


def availableShips(passengerCount):
    """Returns a list of ships that can carry a given number of passengers
    Args:
        passengerCount: the number of passengers
    Returns:
        a list of ships that can carry a given number of passengers
    """
    # Base URL for the Star Wars API (SWAPI) starships endpoint
    url = "https://swapi.dev/api/starships/"
    # Initialize an empty list to store the names of ships that can
    # carry the given number of passengers
    ships = []

    # Loop to paginate through all pages of the API response
    while url:
        # Send a GET request to the current URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code != 200:
            break

        # Parse the JSON response
        data = response.json()
        # Iterate through the list of ships in the current page of the response
        for ship in data.get('results', []):
            try:
                # Get the number of passengers the ship can carry,
                # default to '0' if not available
                passengers = ship.get('passengers', '0').replace(',', '')
                # Check if the passengers value is a digit and if it is
                # greater than or equal to the given passenger count
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    # Add the ship's name to the list of ships
                    ships.append(ship.get('name'))
            except ValueError:
                # If there is a ValueError (e.g., passengers
                # is not a valid number), skip this ship
                continue

        # Get the URL for the next page of results
        url = data.get('next')

    # Return the list of ships that can carry the given number of passengers
    return ships
