import pandas as pd
from datetime import datetime
import json
import ee

json_data = {}

def collect_cloud_probabilities(START_DATE, END_DATE):
    random_point_df = pd.read_csv('data/10k_random.csv')
    coordinates = random_point_df[['lon', 'lat']].values.tolist()
    coordinates = coordinates[:100]

    dataset = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    # filter according to time interval
    filtered_dataset = dataset.filterDate(ee.Date(START_DATE), ee.Date(END_DATE))

    for i, coord in enumerate(coordinates):
        print("Number of point: ", i)
        lon, lat = coord
        point = ee.Geometry.Point(coord)

        # Her koordinat için processed_dates'i ve prob_data'yı sıfırla
        processed_dates = set()
        prob_data = {}

        # filter
        filtered_dataset_ = filtered_dataset.filterBounds(point)

        for image in filtered_dataset_.toList(filtered_dataset_.size()).getInfo():
            image_date = datetime.utcfromtimestamp(image['properties']['system:time_start'] / 1000.0).strftime('%Y-%m-%d')
            print(f"Processing image for date: {image_date}")

            # Check if the date has been processed already, if so, skip to the next image
            if image_date in processed_dates:
                print(f"Skipping image for date {image_date} as it's already processed.")
                continue

            selected_image = ee.Image(image['id'])
            # ... (Diğer işlemleriniz buraya gelecek, örneğin bulut olasılığı hesaplama)

            cloud_image = selected_image.select(['probability'])          
            # Use reduceRegion to get all values, then extract the specific band value
            cloud_dict = cloud_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
            cloud_prob = cloud_dict.get('probability')

            # Eğer bulut olasılığı None ise, veriyi eklemeyin
            if cloud_prob is not None:
                print("Cloud Prob: ", cloud_prob)
                prob_data[image_date] = cloud_prob

            # Add the processed date to the set
            processed_dates.add(image_date)

        # Koordinat ID'sini ve ilgili verileri JSON yapısına ekle
        coord_id = f'{i}'  # Koordinat için benzersiz bir ID oluştur
        json_data[coord_id] = {
            'lon': lon,
            'lat': lat,
            'probabilities': prob_data
        }

    file_path = 'cloud_probabilities.json'

    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)
