import numpy as np
import pandas as pd
import json
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
import scipy.stats as st
import matplotlib.pyplot as plt


def VAR_Model(index):
    print("VAR Model")
    # Read Json
    points_probabilities = 'data/100points_probabilities.json'
    cloud_json = 'data/cloud_probabilities.json'
    cloud_prob = []
    cloud_date = []

    with open(points_probabilities, 'r') as file:
        json_data_probabilities = json.load(file)

    with open(cloud_json, 'r') as file:
        json_cloud = json.load(file)

    # Get class class probabilities
    class1_data = [entry['classes']['class1'] for entry in json_data_probabilities[str(index)]]
    class2_data = [entry['classes']['class2'] for entry in json_data_probabilities[str(index)]]
    class3_data = [entry['classes']['class3'] for entry in json_data_probabilities[str(index)]]
    class4_data = [entry['classes']['class4'] for entry in json_data_probabilities[str(index)]]
    class5_data = [entry['classes']['class5'] for entry in json_data_probabilities[str(index)]]
    class6_data = [entry['classes']['class6'] for entry in json_data_probabilities[str(index)]]

    for key, value in json_cloud[str(index)].items():
        if key == 'probabilities':
            for date, prob in value.items():
                cloud_date.append(date)
                cloud_prob.append(prob)

    # Read csv
    pixel_df4 = pd.read_csv('data/100points_pixelIntensitiesB4.csv')
    pixel_df3 = pd.read_csv('data/100points_pixelIntensitiesB3.csv')
    pixel_df2 = pd.read_csv('data/100points_pixelIntensitiesB2.csv')
    iou_df = pd.read_csv('data/100points_iou.csv')
    scores_df = pd.read_csv('data/100points_scores.csv')

    date_series = pd.to_datetime(cloud_date) 
    date = pd.DataFrame(date_series, columns=['Date'])

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
    cloud_df = pd.DataFrame(cloud_prob, columns=['Cloud'])

    concat = pd.concat([pixel_b4, pixel_b3, pixel_b2, iou, scores,
                        class1_df, class2_df, class3_df, class4_df, class5_df, class6_df, cloud_df], axis=1)

    concat.columns = ['Pixel_Intensity_B4', 'Pixel_Intensity_B3', 'Pixel_Intensity_B2', 
                    'IOU', 'Scores', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Cloud_Prob']

    # concat['Mean_rgb'] = concat[['Pixel_Intensity_B4', 'Pixel_Intensity_B3', 'Pixel_Intensity_B2']].mean(axis=1)
    concat = concat.dropna()

    plt.figure(figsize=(15, 6))

    plt.plot(class1_df, label='class1')
    plt.plot(class2_df, label='class2')
    plt.plot(class3_df, label='class3')
    plt.plot(class4_df, label='class4')
    plt.plot(class5_df, label='class5')
    plt.plot(class6_df, label='class6')
    plt.title('Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()

    plt.tight_layout()
    plt.savefig('graphics/classes.png', dpi=300)


    """
    forecast_ci returns 3 confidence values for each forecasting dates.
    forecast_ci[:, :, :] --> (confidence_values, rows, columns), in this case the shape is usually (3, rows, 11)
    forecast_ci[0:1, :, :] --> returns medium confidence values for all rows and columns
    forecast_ci[1:2, :, :] --> returns the lowest confidence values for all rows and columns
    forecast_ci[2:, :, :] --> --> returns the highest confidence values for all rows and columns
    """

    noise_factor = 0.005
    means = []
    try:
        model = VAR(concat)
        model_fitted = model.fit()

        lag_order = model_fitted.k_ar

        forecast = model_fitted.forecast(concat.values[-lag_order:], steps=12) # forecasting for 12 days                       

        for column in concat.columns:
            means.append(np.mean(concat[column]))

        low_conf, high_conf = st.t.interval(confidence=0.999, df=len(concat)-1, 
                                            loc=means, 
                                            scale=st.sem(concat)) 
        
        # forecast_ci = model_fitted.forecast_interval(concat.values[-lag_order:], steps=12, alpha=0.5)
        # forecast_ci = np.array(forecast_ci)

        # Root Mean Squared Error 
        rmse_result = rmse(concat.values[-lag_order:], forecast) 

        # Akaike Information Criteria
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
        forecast = model_fitted.forecast(concat.values[-lag_order:], steps=12) # forecasting for 12 days

        for column in concat.columns:
            means.append(np.mean(concat[column]))

        low_conf, high_conf = st.t.interval(confidence=0.999, df=len(concat)-1, 
                                            loc=means, 
                                            scale=st.sem(concat)) 

        # Root Mean Squared Error 
        rmse_result = rmse(concat.values[-lag_order:], forecast) 

        # Akaike Information Criteria 
        try:
            aic = model_fitted.aic
        except np.linalg.LinAlgError: # If a value is negative, aic function returns LinAlgError
            aic = None


    print(f"Root Mean Squared Error: {rmse_result}")
    print(f"Akaike Information Criteria AIC: {aic}")

    """
    Concat and forecast dataframes are merged.
    Date, Fitted and Index columns are added.
    Fitted 0 --> It means that, it's row used in VAR Model
    Fitted 1 --> It means that, it's row is forecasting data
    """

    forecast_df_ = pd.DataFrame(forecast, columns=concat.columns)
    forecast_df_['Fitted'] = 1
    datetime_series = pd.to_datetime(date['Date'])  
    begining_date = datetime_series.max()

    for i in range(len(forecast_df_)):
        begining_date += pd.Timedelta(days=1)
        forecast_df_.at[i, 'Date'] = begining_date.date()

    full_df = concat.copy()
    full_df['Date'] = date 
    full_df['Fitted'] = 0
    full_df = pd.concat([full_df, forecast_df_], axis=0)

    full_df['Index'] = range(1, len(full_df) + 1)
    full_df['B4_CI'] = full_df.apply(lambda row: 1 if row['Pixel_Intensity_B4'] >= low_conf[0] and row['Pixel_Intensity_B4'] <= high_conf[0] else 0, axis=1)
    full_df['B3_CI'] = full_df.apply(lambda row: 1 if row['Pixel_Intensity_B3'] >= low_conf[1] and row['Pixel_Intensity_B3'] <= high_conf[1] else 0, axis=1)
    full_df['B2_CI'] = full_df.apply(lambda row: 1 if row['Pixel_Intensity_B2'] >= low_conf[2] and row['Pixel_Intensity_B2'] <= high_conf[2] else 0, axis=1)
    full_df['IOU_CI'] = full_df.apply(lambda row: 1 if row['IOU'] >= low_conf[3] and row['IOU'] <= high_conf[3] else 0, axis=1)
    full_df['Scores_CI'] = full_df.apply(lambda row: 1 if row['Scores'] >= low_conf[4] and row['Scores'] <= high_conf[4] else 0, axis=1)
    full_df['Class1_CI'] = full_df.apply(lambda row: 1 if row['Class1'] >= low_conf[5] and row['Class1'] <= high_conf[5] else 0, axis=1)
    full_df['Class2_CI'] = full_df.apply(lambda row: 1 if row['Class2'] >= low_conf[6] and row['Class2'] <= high_conf[6] else 0, axis=1)
    full_df['Class3_CI'] = full_df.apply(lambda row: 1 if row['Class3'] >= low_conf[7] and row['Class3'] <= high_conf[7] else 0, axis=1)
    full_df['Class4_CI'] = full_df.apply(lambda row: 1 if row['Class4'] >= low_conf[8] and row['Class4'] <= high_conf[8] else 0, axis=1)
    full_df['Class5_CI'] = full_df.apply(lambda row: 1 if row['Class5'] >= low_conf[9] and row['Class5'] <= high_conf[9] else 0, axis=1)
    full_df['Class6_CI'] = full_df.apply(lambda row: 1 if row['Class6'] >= low_conf[10] and row['Class6'] <= high_conf[10] else 0, axis=1)
    full_df['Cloud_Prob_CI'] = full_df.apply(lambda row: 1 if row['Cloud_Prob'] >= low_conf[10] and row['Cloud_Prob'] <= high_conf[10] else 0, axis=1)
    full_df.to_csv('data/example_result_for_a_point.csv')

    try:
        percentage_1 = (full_df['Class2_CI'].value_counts(normalize=True) * 100).loc[1]
        print(f"Percentage of 1s: {percentage_1}%")
    except KeyError:
        print(f"Percentage of 1s: {0}%")

    """
    forecast_ci[1:2, :, column_number].flatten(), forecast_ci[2:, :, column_number].flatten() --> alternative confidence interval
    concatenated_df['Pixel_Intensity_B4'].iloc[-len(forecast):].index
    """
    forecast_df = pd.DataFrame(forecast, columns=concat.columns)
    concatenated_df = pd.concat([concat, forecast_df], ignore_index=True)

    plt.figure(figsize=(15, 6))

    plt.plot(concatenated_df['Pixel_Intensity_B4'], label='Pixel_Intensity_B4 Red')
    plt.plot(concatenated_df['Pixel_Intensity_B4'].iloc[:-len(forecast)], label='Historical Pixel_Intensity')
    plt.plot(concatenated_df['Pixel_Intensity_B4'].iloc[-len(forecast):], label='Pixel_Intensity Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Pixel_Intensity_B4'].index, low_conf[0], high_conf[0], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Pixel_Intensity B4')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/pixel_intensity_B4.png', dpi=300)

    plt.plot(concatenated_df['Pixel_Intensity_B3'], label='Pixel_Intensity_B3 Green')
    plt.plot(concatenated_df['Pixel_Intensity_B3'].iloc[:-len(forecast)], label='Historical Pixel_Intensity')
    plt.plot(concatenated_df['Pixel_Intensity_B3'].iloc[-len(forecast):], label='Pixel_Intensity Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Pixel_Intensity_B3'].index, low_conf[1], high_conf[1], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Pixel_Intensity B3')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/pixel_intensity_B3.png', dpi=300)

    plt.plot(concatenated_df['Pixel_Intensity_B2'], label='Pixel_Intensity_B2 Blue')
    plt.plot(concatenated_df['Pixel_Intensity_B2'].iloc[:-len(forecast)], label='Historical Pixel_Intensity')
    plt.plot(concatenated_df['Pixel_Intensity_B2'].iloc[-len(forecast):], label='Pixel_Intensity Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Pixel_Intensity_B2'].index, low_conf[2], high_conf[2], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Pixel_Intensity B2')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/pixel_intensity_B2.png', dpi=300)

    plt.figure(figsize=(15, 6))

    plt.plot(concatenated_df['IOU'], label='IOU Real')
    plt.plot(concatenated_df['IOU'].iloc[:-len(forecast)], label='Historical IOU')
    plt.plot(concatenated_df['IOU'].iloc[-len(forecast):], label='IOU Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['IOU'].index, low_conf[3], high_conf[3], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - IOU')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/IOU.png', dpi=300)

    plt.subplot(122)
    plt.plot(concatenated_df['Scores'], label='Scores Real')
    plt.plot(concatenated_df['Scores'].iloc[:-len(forecast)], label='Historical Scores')
    plt.plot(concatenated_df['Scores'].iloc[-len(forecast):], label='Scores Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Scores'].index, low_conf[4], high_conf[4], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Scores')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Scores.png', dpi=300)

    plt.figure(figsize=(15, 6))

    plt.plot(concatenated_df['Class1'], label='Class1')
    plt.plot(concatenated_df['Class1'].iloc[:-len(forecast)], label='Historical Class1 Probabilities')
    plt.plot(concatenated_df['Class1'].iloc[-len(forecast):], label='Class1 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class1'].index, low_conf[5], high_conf[5], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class1')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class1.png', dpi=300)

    plt.plot(concatenated_df['Class2'], label='Class2')
    plt.plot(concatenated_df['Class2'].iloc[:-len(forecast)], label='Historical Class2 Probabilities')
    plt.plot(concatenated_df['Class2'].iloc[-len(forecast):], label='Class2 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class2'].index, low_conf[6], high_conf[6], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class2')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class2.png', dpi=300)

    plt.plot(concatenated_df['Class3'], label='Class3')
    plt.plot(concatenated_df['Class3'].iloc[:-len(forecast)], label='Historical Class3 Probabilities')
    plt.plot(concatenated_df['Class3'].iloc[-len(forecast):], label='Class3 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class3'].index, low_conf[7], high_conf[7], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class3')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class3.png', dpi=300)

    plt.plot(concatenated_df['Cloud_Prob'], label='Cloud')
    plt.plot(concatenated_df['Cloud_Prob'].iloc[:-len(forecast)], label='Historical Cloud Probabilities')
    plt.plot(concatenated_df['Cloud_Prob'].iloc[-len(forecast):], label='Cloud Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Cloud_Prob'].index, low_conf[11], high_conf[11], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Cloud')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Cloud_Prob.png', dpi=300)

    plt.figure(figsize=(15, 6))

    plt.plot(concatenated_df['Class4'], label='Class1')
    plt.plot(concatenated_df['Class4'].iloc[:-len(forecast)], label='Historical Class4 Probabilities')
    plt.plot(concatenated_df['Class4'].iloc[-len(forecast):], label='Class4 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class4'].index, low_conf[8], high_conf[8], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class4')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class4.png', dpi=300)

    plt.plot(concatenated_df['Class5'], label='Class5')
    plt.plot(concatenated_df['Class5'].iloc[:-len(forecast)], label='Historical Class5 Probabilities')
    plt.plot(concatenated_df['Class5'].iloc[-len(forecast):], label='Class5 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class5'].index, low_conf[9], high_conf[9], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class5')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class5.png', dpi=300)

    plt.plot(concatenated_df['Class6'], label='Class6')
    plt.plot(concatenated_df['Class6'].iloc[:-len(forecast)], label='Historical Class6 Probabilities')
    plt.plot(concatenated_df['Class6'].iloc[-len(forecast):], label='Class6 Prediction', linestyle='dashed')
    plt.fill_between(concatenated_df['Class6'].index, low_conf[10], high_conf[10], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title('VAR Model - Class6')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('graphics/Class6.png', dpi=300)
