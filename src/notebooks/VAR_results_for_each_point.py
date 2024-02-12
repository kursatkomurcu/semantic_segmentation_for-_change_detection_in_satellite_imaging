import numpy as np
import pandas as pd
import json
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse

noise_factor = 0.01

# Read csv
pixel_df4 = pd.read_csv('data/100points_pixelIntensitiesB4.csv')
pixel_df3 = pd.read_csv('data/100points_pixelIntensitiesB3.csv')
pixel_df2 = pd.read_csv('data/100points_pixelIntensitiesB2.csv')
iou_df = pd.read_csv('data/100points_iou.csv')
scores_df = pd.read_csv('data/100points_scores.csv')

# Read Json
points_probabilities = 'data/100points_probabilities.json'
cloud_json = 'data/cloud_probabilities.json'
cloud_prob = []
cloud_date = []

with open(points_probabilities, 'r') as file:
    json_data_probabilities = json.load(file)

with open(cloud_json, 'r') as file:
    json_cloud = json.load(file)

data = {}

for index in range(100):
    print(index)
    class1_data = [entry['classes']['class1'] for entry in json_data_probabilities[str(index)]]
    class2_data = [entry['classes']['class2'] for entry in json_data_probabilities[str(index)]]
    class3_data = [entry['classes']['class3'] for entry in json_data_probabilities[str(index)]]
    class4_data = [entry['classes']['class4'] for entry in json_data_probabilities[str(index)]]
    class5_data = [entry['classes']['class5'] for entry in json_data_probabilities[str(index)]]
    class6_data = [entry['classes']['class6'] for entry in json_data_probabilities[str(index)]]

    pixel_b4 = pixel_df4[[str(index)]]
    pixel_b3 = pixel_df3[[str(index)]]
    pixel_b2 = pixel_df2[[str(index)]]
    iou = iou_df[[str(index)]]
    scores = scores_df[[str(index)]]
    class1_df = pd.DataFrame(class1_data)
    class2_df = pd.DataFrame(class2_data)
    class3_df = pd.DataFrame(class3_data)
    class4_df = pd.DataFrame(class4_data)
    class5_df = pd.DataFrame(class5_data)
    class6_df = pd.DataFrame(class6_data)

    for key, value in json_cloud[str(index)].items():
        if key == 'probabilities':
            for date, prob in value.items():
                cloud_date.append(date)
                cloud_prob.append(prob)
    
    date_series = pd.to_datetime(cloud_date) 
    date = pd.DataFrame(date_series, columns=['Date'])
    cloud_df = pd.DataFrame(cloud_prob, columns=['Cloud'])

    concat = pd.concat([pixel_b4, pixel_b3, pixel_b2, iou, scores,
                    class1_df, class2_df, class3_df, class4_df, class5_df, class6_df, cloud_df], axis=1)

    concat.columns = ['Pixel_Intensity_B4', 'Pixel_Intensity_B3', 'Pixel_Intensity_B2', 
                  'IOU', 'Scores', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Cloud_Prob']


    concat = concat.dropna()

    # VAR Model
    try:
        model = VAR(concat)
        model_fitted = model.fit()

        lag_order = model_fitted.k_ar
        forecast = model_fitted.forecast(concat.values[-lag_order:], steps=12)

        forecast_ci = model_fitted.forecast_interval(concat.values[-lag_order:], steps=12, alpha=0.05)
        forecast_ci = np.array(forecast_ci)

        # Root Mean Squared Error (data type --> float)
        rmse_result = rmse(concat.values[-lag_order:], forecast) 

        # Akaike Information Criteria (data type --> float)
        try:
            aic = model_fitted.aic
        except np.linalg.LinAlgError: # If a value is negative, aic function returns LinAlgError
            aic = None

    except ValueError: 
        '''
        If a column is constant, VAR model doesn't work. 
        To prevent this situation small amount of noise is added to columns
        '''
        for column in concat.columns:
            noise = np.random.normal(0, noise_factor, concat.shape[0])
            concat[column] = concat[column] + noise
        
        model = VAR(concat)
        model_fitted = model.fit()

        lag_order = model_fitted.k_ar
        forecast = model_fitted.forecast(concat.values[-lag_order:], steps=12)

        forecast_ci = model_fitted.forecast_interval(concat.values[-lag_order:], steps=12, alpha=0.05)
        forecast_ci = np.array(forecast_ci)

        # Root Mean Squared Error (data type --> float)
        rmse_result = rmse(concat.values[-lag_order:], forecast) 

        # Akaike Information Criteria (data type --> float)
        try:
            aic = model_fitted.aic
        except np.linalg.LinAlgError: # If a value is negative, aic function returns LinAlgError
            aic = None

    # Confidence Values (data type --> 1d array)
    conf_mid_b4 = forecast_ci[0:1, :, 0].flatten()
    conf_lower_b4 = forecast_ci[1:2, :, 0].flatten()
    conf_upper_b4 = forecast_ci[2:, :, 0].flatten()

    conf_mid_b3 = forecast_ci[0:1, :, 1].flatten()
    conf_lower_b3 = forecast_ci[1:2, :, 1].flatten()
    conf_upper_b3 = forecast_ci[2:, :, 1].flatten()

    conf_mid_b2 = forecast_ci[0:1, :, 2].flatten()
    conf_lower_b2 = forecast_ci[1:2, :, 2].flatten()
    conf_upper_b2 = forecast_ci[2:, :, 2].flatten()

    conf_mid_iou = forecast_ci[0:1, :, 3].flatten()
    conf_lower_iou = forecast_ci[1:2, :, 3].flatten()
    conf_upper_iou = forecast_ci[2:, :, 3].flatten()

    conf_mid_score = forecast_ci[0:1, :, 4].flatten()
    conf_lower_score = forecast_ci[1:2, :, 4].flatten()
    conf_upper_score = forecast_ci[2:, :, 4].flatten()

    conf_mid_class1 = forecast_ci[0:1, :, 5].flatten()
    conf_lower_class1 = forecast_ci[1:2, :, 5].flatten()
    conf_upper_class1 = forecast_ci[2:, :, 5].flatten()

    conf_mid_class2 = forecast_ci[0:1, :, 6].flatten()
    conf_lower_class2 = forecast_ci[1:2, :, 6].flatten()
    conf_upper_class2 = forecast_ci[2:, :, 6].flatten()

    conf_mid_class3 = forecast_ci[0:1, :, 7].flatten()
    conf_lower_class3 = forecast_ci[1:2, :, 7].flatten()
    conf_upper_class3 = forecast_ci[2:, :, 7].flatten()

    conf_mid_class4 = forecast_ci[0:1, :, 8].flatten()
    conf_lower_class4 = forecast_ci[1:2, :, 8].flatten()
    conf_upper_class4 = forecast_ci[2:, :, 8].flatten()

    conf_mid_class5 = forecast_ci[0:1, :, 9].flatten()
    conf_lower_class5 = forecast_ci[1:2, :, 9].flatten()
    conf_upper_class5 = forecast_ci[2:, :, 9].flatten()

    conf_mid_class6 = forecast_ci[0:1, :, 10].flatten()
    conf_lower_class6 = forecast_ci[1:2, :, 10].flatten()
    conf_upper_class6 = forecast_ci[2:, :, 10].flatten()

    conf_mid_cloud = forecast_ci[0:1, :, 11].flatten()
    conf_lower_cloud = forecast_ci[1:2, :, 11].flatten()
    conf_upper_cloud = forecast_ci[2:, :, 11].flatten()

    # print(rmse_result)
    temp_data = [
        {
            'rmse_results': {
                'rmse_b4': rmse_result[0], 'rmse_b3': rmse_result[1], 'rmse_b2': rmse_result[2],
                'rmse_iou': rmse_result[3], 'rmse_score': rmse_result[4],
                'rmse_class1': rmse_result[5], 'rmse_class2': rmse_result[6], 'rmse_class3': rmse_result[7],
                'rmse_class4': rmse_result[8], 'rmse_class5': rmse_result[9], 'rmse_class6': rmse_result[10],
                'rmse_cloud' : rmse_result[11]
            },
            'aic': aic,
            'confidence_intervals': {
                'conf_mid_b4': conf_mid_b4.tolist(), 'conf_lower_b4': conf_lower_b4.tolist(), 'conf_upper_b4': conf_upper_b4.tolist(),
                'conf_mid_b3': conf_mid_b3.tolist(), 'conf_lower_b3': conf_lower_b3.tolist(), 'conf_upper_b3': conf_upper_b3.tolist(),
                'conf_mid_b2': conf_mid_b2.tolist(), 'conf_lower_b2': conf_lower_b2.tolist(), 'conf_upper_b2': conf_upper_b2.tolist(),
                'conf_mid_iou': conf_mid_iou.tolist(), 'conf_lower_iou': conf_lower_iou.tolist(), 'conf_upper_iou': conf_upper_iou.tolist(),
                'conf_mid_score': conf_mid_score.tolist(), 'conf_lower_score': conf_lower_score.tolist(), 'conf_upper_score': conf_upper_score.tolist(),
                'conf_mid_class1': conf_mid_class1.tolist(), 'conf_lower_class1': conf_lower_class1.tolist(), 'conf_upper_class1': conf_upper_class1.tolist(),
                'conf_mid_class2': conf_mid_class2.tolist(), 'conf_lower_class2': conf_lower_class2.tolist(), 'conf_upper_class2': conf_upper_class2.tolist(),
                'conf_mid_class3': conf_mid_class3.tolist(), 'conf_lower_class3': conf_lower_class3.tolist(), 'conf_upper_class3': conf_upper_class3.tolist(),
                'conf_mid_class4': conf_mid_class4.tolist(), 'conf_lower_class4': conf_lower_class4.tolist(), 'conf_upper_class4': conf_upper_class4.tolist(),
                'conf_mid_class5': conf_mid_class5.tolist(), 'conf_lower_class5': conf_lower_class5.tolist(), 'conf_upper_class5': conf_upper_class5.tolist(),
                'conf_mid_class6': conf_mid_class6.tolist(), 'conf_lower_class6': conf_lower_class6.tolist(), 'conf_upper_class6': conf_upper_class6.tolist(),
                'conf_mid_cloud': conf_mid_cloud.tolist(), 'conf_lower_cloud': conf_lower_cloud.tolist(), 'conf_upper_cloud': conf_upper_cloud.tolist()
            }
        }
    ]

    data[index] = temp_data

with open('data/VAR_results.json', 'w') as file:
    json.dump(data, file)

