#!/usr/bin/env python3
"""This module uses the SWAPI API to return a list of
homeworld planets of all sentient species."""
import requests


def sentientPlanets():
    """
    Fetch a list of homeworld planets of all sentient species.
    Maintains the order provided by the SWAPI API.
    Returns:
        A list of homeworld planets of all sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets_list = []

    while url:
        # Send a GET request to the species URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data: {response.status_code}")

        # Parse the JSON response
        data = response.json()

        # Iterate through each species in the results
        for species in data.get("results", []):
            # Check if the species is sentient
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                # Fetch the homeworld URL
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    # Send a GET request to the homeworld URL
                    homeworld_response = requests.get(homeworld_url)

                    # Check if the request was successful
                    if homeworld_response.status_code == 200:
                        # Get the homeworld name from the JSON response
                        homeworld_name = homeworld_response.json().get("name")

                        # Only add the homeworld name to the list if it exists
                        if homeworld_name:
                            planets_list.append(homeworld_name)

        # Get the URL for the next page of species
        url = data.get("next")

    return planets_list
