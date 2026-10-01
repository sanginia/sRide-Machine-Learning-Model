import pandas as pd

ride_requests = pd.read_csv("data/ride_requests.csv")
users = pd.read_csv("data/users.csv")

print("Ride requests shape:", ride_requests.shape)
print("Users shape:", users.shape)

print("\nRide request columns:")
print(ride_requests.columns.tolist())

print("\nUsers columns:")
print(users.columns.tolist())

print("\nCompletion label distribution:")
print(ride_requests["completed_label"].value_counts(normalize=True))

print("\nFirst few ride request rows:")
print(ride_requests.head())