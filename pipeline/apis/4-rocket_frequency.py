#!/usr/bin/env python3
"""This module fetches and displays the number of launches per rocket."""
import requests
from collections import defaultdict


def get_launches_per_rocket():
    """
    Fetch and display the number of launches per rocket.
    """
    # SpaceX API URL for launches
    url = "https://api.spacexdata.com/v4/launches"

    # Send a GET request to the SpaceX API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    # Parse the JSON response
    launches = response.json()

    # Dictionary to store the count of launches per rocket
    rocket_launch_count = defaultdict(int)

    # Count the number of launches per rocket
    for launch in launches:
        rocket_id = launch["rocket"]
        rocket_launch_count[rocket_id] += 1

    # Fetch rocket names
    rocket_url = "https://api.spacexdata.com/v4/rockets"
    rocket_response = requests.get(rocket_url)
    rockets = rocket_response.json()

    # Dictionary to map rocket IDs to rocket names
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    # List to store the rocket names and their launch counts
    rocket_launch_list = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_launch_count.items()
    ]

    # Sort the list by the number of launches (descending)
    # and then by rocket name (alphabetic order)
    rocket_launch_list.sort(key=lambda x: (-x[1], x[0]))

    # Print the result
    for rocket_name, count in rocket_launch_list:
        print(f"{rocket_name}: {count}")


if __name__ == "__main__":
    get_launches_per_rocket()
