#!/usr/bin/env python3
"""
This mdoule fetches the location of a GitHub user from the GitHub API.
"""
import requests
import sys
from datetime import datetime


def get_user_location(api_url):
    """
    Fetch the location of a GitHub user from the GitHub API.

    If the user doesn't exist (404), print 'Not found'.
    If rate-limited (403), print the time to reset in minutes.
    If the user exists, print the user's location or 'No location found'.
    """
    response = requests.get(api_url)

    # If user does not exist (404)
    if response.status_code == 404:
        print("Not found")

    # If rate limit is exceeded (403)
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-RateLimit-Reset'))
        current_time = int(datetime.utcnow().timestamp())
        reset_in_minutes = (reset_time - current_time) // 60
        print(f"Reset in {reset_in_minutes} min")

    # If the user is found (200)
    elif response.status_code == 200:
        user_data = response.json()
        location = user_data.get("location")

        if location:
            print(location)
        else:
            print("No location found")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
