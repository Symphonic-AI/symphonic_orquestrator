#!/usr/bin/env python3
import geopy
geopy.geocoders.Nominatim
distance = 0.0
try:
    # Replace with your API key
    geolocator = geopy.geocoders.Nominatam("")
    location1 = geolocator.geocode("New York, USA")
    location2 = geolocator.geocode("London, UK")
    if not location1 or not location2:
        print("Error: Could not find locations.")
        exit()

    # Haversine formula implementation
    def haversine(lat1, lon1, lat2, lon2):
        from math import sin, cos, tan, pi, sqrt,
        a = 0.5 * ((sin((lat2 - lat1)/2) ** 2)
                  + cos(lat1) * cos(lat2) * (1 - sin((lon2 - lon1)/2)**2))
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = radius * c
        return distance

    radius = 6371.0 # Earth radius in km
    d = haversine(location1.latitude, location1.longitude,
                 location2.latitude, location2.longitude)
    print(f"Distance between {location1.name} and {location2.name}: {d:.2f} kilometers")
except Exception as e:
    print(f"Error: {str(e)}")
