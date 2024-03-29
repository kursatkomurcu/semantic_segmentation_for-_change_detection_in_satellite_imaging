import random
import pandas as pd

def generate_random_coordinates(num_points, min_lat, max_lat, min_lon, max_lon):
    data_set = []
    
    for _ in range(num_points):
        latitude = random.uniform(min_lat, max_lat)
        longitude = random.uniform(min_lon, max_lon)
        data_set.append((latitude, longitude))
    
    return data_set

def create_dataset_with_ids(coordinates):
    dataset_with_ids = [(lat, lon, idx) for idx, (lat, lon) in enumerate(coordinates)]
    return dataset_with_ids


def create_dataset_with_random_coordinates():
    print("Creating a dataset with random points between given interval")
    # Coordinate intervals
    min_latitude = 53.53100
    max_latitude = 59.69747
    min_longitude = 20.49722
    max_longitude = 28.22760

    # Random 10000 points
    random_coordinates = generate_random_coordinates(10000, min_latitude, max_latitude, min_longitude, max_longitude)

    # Create dataset and set id
    dataset_with_ids = create_dataset_with_ids(random_coordinates)

    for lat, lon, idx in dataset_with_ids:
        print(f"latitude: {lat}, longitude: {lon}, id: {idx}")

    df = pd.DataFrame(dataset_with_ids, columns=['lat', 'lon', 'id'])
    df.to_csv('10k_random.csv', index=False)
    
    print("Creating dataset process is finished")
