import ee
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from samgeo.common import *
from samgeo.samgeo import *
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import cv2
import tempfile

def collect_iou(start_date, end_date):
  print("Collecting IOU")
  sam = SamGeo(
      model_type="vit_h",
      checkpoint="sam_vit_h_4b8939.pth",
      sam_kwargs=None,
      automatic = True, # --> to use generate function set automatic param as True,
  )                      # --> to use predict function set automatic param as False

  random_point_df = pd.read_csv('10k_random.csv')
  coordinates = random_point_df[['lon', 'lat']].values.tolist()

  dataset = ee.ImageCollection('COPERNICUS/S2')

  # filter according to time interval
  filtered_dataset = dataset.filterDate(ee.Date(start_date), ee.Date(end_date))

  df = pd.DataFrame()

  ious = []
  for i in range(len(coordinates)):
    if i >= 21:
      print("Number of point: ", i)
      point = ee.Geometry.Point(coordinates[i])

      # filter
      filtered_dataset_ = filtered_dataset.filterBounds(point)

      flag = False
      for image in filtered_dataset_.toList(filtered_dataset_.size()).getInfo():
        image_date = datetime.utcfromtimestamp(image['properties']['system:time_start'] / 1000.0).strftime('%Y-%m-%d')
        print(f"Processing image for date: {image_date}")

        selected_image = ee.Image(image['id'])
        rgb_image = selected_image.select(['B4', 'B3', 'B2'])

        url = rgb_image.getThumbURL({'dimensions': 256, 'format': 'png'})

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Resize the image to meet the library's requirements
        resized_image = cv2.resize(np.array(img), (256, 256))

        # Convert the NumPy array to a PyTorch tensor
        tensor_image = transforms.ToTensor()(resized_image)

        # Add a batch dimension (BCHW)
        tensor_image = tensor_image.unsqueeze(0) # (batch, channel, width, height) --> increase batch

        # create df
        if not flag:
          print(f"IOU Date: {image_date}")
          with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img_path = temp_img.name
            temp_img.close()
            transforms.ToPILImage()(tensor_image.squeeze()).save(temp_img_path) # tensor_image

            # Call sam.generate with the file path
            sam.generate(temp_img_path, output="masks.png", foreground=True, unique=True)

          pred_iou = sam.probability(axis="on", alpha=1, output="annotations.png")
          df = pd.concat([df, pd.DataFrame({'Date_'+ str(i): [image_date],  str(i) : [pred_iou]}),], ignore_index=True) # df = pd.concat([df, pd.DataFrame({'Date_'+ str(i): [image_date],  str(i) : [pred_iou]}),], ignore_index=True)

          os.remove(temp_img_path)
          flag = True

        if (image_date not in df['Date_' + str(i)].values):
          print(f"IOU Date: {image_date}")
          with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img_path = temp_img.name
            temp_img.close()
            transforms.ToPILImage()(tensor_image.squeeze()).save(temp_img_path)

            # Call sam.generate with the file path
            sam.generate(temp_img_path, output="masks.png", foreground=True, unique=True)

          pred_iou = sam.probability(axis="on", alpha=1, output="annotations.png")
          df = pd.concat([df, pd.DataFrame({'Date_'+ str(i): [image_date],  str(i) : [pred_iou]}),], ignore_index=True)

          # Remove the temporary image file
          os.remove(temp_img_path)

      ious.append(df)
      df = pd.DataFrame()
      if i != 0 and i % 10 == 0:
        save_df = pd.concat(ious, axis=1)
        save_df.to_csv('100points_iou.csv')

  save_df = pd.concat(ious, axis=1)
  save_df.to_csv('100points_iou.csv')
  print("Collecting IOU finished")

def collect_scores(start_date, end_date):
  print("Scores Collecting")
  random_point_df = pd.read_csv('10k_random.csv')
  coordinates = random_point_df[['lon', 'lat']].values.tolist()

  dataset = ee.ImageCollection('COPERNICUS/S2')

  # filter according to time interval
  filtered_dataset = dataset.filterDate(ee.Date(start_date), ee.Date(end_date))


  sam = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    sam_kwargs=None,
    automatic = False, # --> to use generate function set automatic param as True,
  ) 

  df = pd.DataFrame()

  ious = []
  for i in range(len(coordinates)):
    if i >= 21:
      print("Number of point: ", i)
      point = ee.Geometry.Point(coordinates[i])

      # filter
      filtered_dataset_ = filtered_dataset.filterBounds(point)

      flag = False
      for image in filtered_dataset_.toList(filtered_dataset_.size()).getInfo():
        image_date = datetime.utcfromtimestamp(image['properties']['system:time_start'] / 1000.0).strftime('%Y-%m-%d')
        print(f"Processing image for date: {image_date}")

        selected_image = ee.Image(image['id'])
        rgb_image = selected_image.select(['B4', 'B3', 'B2'])

        url = rgb_image.getThumbURL({'dimensions': 256, 'format': 'png'})

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Resize the image to meet the library's requirements
        resized_image = cv2.resize(np.array(img), (256, 256))

        # Convert the NumPy array to a PyTorch tensor
        tensor_image = transforms.ToTensor()(resized_image)

        # Add a batch dimension (BCHW)
        tensor_image = tensor_image.unsqueeze(0) # (batch, channel, width, height) --> increase batch

        # create df
        if not flag:
          print(f"IOU Date: {image_date}")
          with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img_path = temp_img.name
            temp_img.close()
            transforms.ToPILImage()(tensor_image.squeeze()).save(temp_img_path)

            sam.set_image(temp_img_path)
            point_ = [[coordinates[i][1], coordinates[i][0]]]

            masks, scores, logits = sam.predict(point_coords=point_, return_results=True, multimask_output=False, output="mask2.tif")

            print("Scores: ", scores)

          df = pd.concat([df, pd.DataFrame({'Date_' + str(i): [image_date], str(i): [scores]}), ], ignore_index=True)

          os.remove(temp_img_path)

          flag = True

        if (image_date not in df['Date_' + str(i)].values):
          print(f"IOU Date: {image_date}")
          with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img_path = temp_img.name
            temp_img.close()
            transforms.ToPILImage()(tensor_image.squeeze()).save(temp_img_path)

            sam.set_image(temp_img_path)
            point_ = [[coordinates[i][1], coordinates[i][0]]]

            masks, scores, logits = sam.predict(point_coords=point_, return_results=True, multimask_output=False, output="mask2.tif")

            print("Scores: ", scores)

          df = pd.concat([df, pd.DataFrame({'Date_' + str(i): [image_date], str(i): [scores]}), ], ignore_index=True)

          os.remove(temp_img_path)

      ious.append(df)
      df = pd.DataFrame()
      if i != 0 and i % 10 == 0:
        save_df = pd.concat(ious, axis=1)
        save_df.to_csv('100points_scores.csv')

  save_df = pd.concat(ious, axis=1)
  save_df.to_csv('100points_scores.csv')
  print("Scores Collecting finished")

