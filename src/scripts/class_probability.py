import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout

import requests
from io import BytesIO
import torchvision.transforms as transforms
import json

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_inception_resnetv2_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained InceptionResNetV2 Model """
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = encoder.get_layer("input_1").output           ## (512 x 512)

    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
    
    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(6, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    return model

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def load_model():
    K.clear_session()

    model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
    model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])
    model.summary()
    model.load_weights("InceptionResNetV2-UNet.h5")

    return model

def class_probabilities(start_date, end_date):
    random_point_df = pd.read_csv('10k_random.csv')
    coordinates = random_point_df[['lon', 'lat']].values.tolist()
    coordinates = coordinates[:100]

    dataset = ee.ImageCollection('COPERNICUS/S2')

    # filter according to time interval
    filtered_dataset = dataset.filterDate(ee.Date(start_date), ee.Date(end_date))

    model = load_model()

    data = {}

    for i in range(len(coordinates)):
        print("Number of point: ", i)
        temp_data = []
        point = ee.Geometry.Point(coordinates[i])

        processed_dates = set()

        # filter
        filtered_dataset_ = filtered_dataset.filterBounds(point)

        for image in filtered_dataset_.toList(filtered_dataset_.size()).getInfo():
            image_date = datetime.utcfromtimestamp(image['properties']['system:time_start'] / 1000.0).strftime('%Y-%m-%d')
            print(f"Processing image for date: {image_date}")

            if image_date in processed_dates:
                print(f"Skipping image for date {image_date} as it's already processed.")
                continue

            selected_image = ee.Image(image['id'])
            rgb_image = selected_image.select(['B4', 'B3', 'B2'])

            url = rgb_image.getThumbURL({'dimensions': 256, 'format': 'png'})

            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            img = img.convert("RGB")  

            input_shape = model.input_shape[1:3]
            img = img.resize(input_shape)
            image_array = np.array(img) / 255.0 

            print(f"Probability Date: {image_date}")
            prediction = model.predict(np.expand_dims(image_array, axis=0))
                
            batch, height, width, channel = prediction.shape
            center_height = height // 2
            center_width = width // 2
            center_values = prediction[:, center_height, center_width, :]
            print(center_values)

            class_probs = {
                "class1" : float(center_values[0][0]),
                "class2" : float(center_values[0][1]),
                "class3" : float(center_values[0][2]),
                "class4" : float(center_values[0][3]),
                "class5" : float(center_values[0][4]),
                "class6" : float(center_values[0][5]),
            }

            new_data = {
                "date" : image_date, 
                "lon" : float(coordinates[i][0]),
                "lat" : float(coordinates[i][1]),
                "classes" : class_probs,  
            }

            processed_dates.add(image_date)

            temp_data.append(new_data)

        data[i] = temp_data


    with open('100points_probabilities.json', 'w') as file:
        json.dump(data, file)