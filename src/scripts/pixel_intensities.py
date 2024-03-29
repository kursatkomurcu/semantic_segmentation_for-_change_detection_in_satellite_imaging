import ee
from datetime import datetime
import pandas as pd

def collect_pixel_intensities(start_date, end_date, bands):
    print("Pixel Intensities are collecting")

    random_point_df = pd.read_csv('data/10k_random.csv')
    coordinates = random_point_df[['lon', 'lat']].values.tolist()
    coordinates = coordinates[:100]

    dataset = ee.ImageCollection('COPERNICUS/S2')

    # filter according to time interval
    filtered_dataset = dataset.filterDate(ee.Date(start_date), ee.Date(end_date))

    data_list = []
    df_result_list = []
    # bands = ['B4', 'B3', 'B2']

    for band in bands:
        for i, coord in enumerate(coordinates):
            print(f"Number of point: {i}")
            point = ee.Geometry.Point(coord)
            filtered_dataset_ = filtered_dataset.filterBounds(point)
            
            data_list = []
            processed_dates = set()
            
            for image_info in filtered_dataset_.toList(filtered_dataset_.size()).getInfo():
                image_date = datetime.utcfromtimestamp(image_info['properties']['system:time_start'] / 1000.0).strftime('%Y-%m-%d')
                
                if image_date in processed_dates:
                    print(f"Skipping image for date {image_date} as it's already processed.")
                    continue
                
                processed_dates.add(image_date)
                print(f"Processing image for date: {image_date}")
                
                selected_image = ee.Image(image_info['id'])
                rgb_image = selected_image.select(['B4', 'B3', 'B2'])
                pixel_intensity_dict = rgb_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
                pixel_intensity = pixel_intensity_dict.get(band)
                
                data_list.append({
                    'Date_' + str(i): image_date,
                    'pixel_intensity': pixel_intensity
                })

            df_result_list.append(pd.DataFrame(data_list))
            
            # Save after every 10th point
            if i % 10 == 0:
                save_df = pd.concat(df_result_list, axis=1)
                save_df.to_csv(f'100points_pixelIntensities{band}.csv')

        # Handle remaining data frames
        if df_result_list:
            save_df = pd.concat(df_result_list, axis=1)
            save_df.to_csv(f'100points_pixelIntensities{band}.csv')
    
    print("Pixel Intensities were collected")