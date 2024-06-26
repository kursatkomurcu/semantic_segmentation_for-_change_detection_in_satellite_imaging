# Semantic Segmentation For Change Detection In Satellite Imaging

In the study we investigate the estimation of non-changing temporal situations in satellite imagery. The publication propose the addition of additional semantic information usage for tracking changes. The raw and semantic information modelled by vector auto-regressive models. The experiments demonstrated succesfull usage of the method

## Links:
1- https://github.com/opengeos/segment-geospatial/tree/main

2- https://github.com/ayushdabra/dubai-satellite-imagery-segmentation/tree/main

3- dataspace.copernicus.eu

## Datasets:
**100points_pixelIntensitiesB4.csv** --> it contains red color pixel intensity values each 100 points for a year 

**100points_pixelIntensitiesB3.csv** --> it contains green color pixel intensity values each 100 points for a year 

**100points_pixelIntensitiesB2.csv** --> it contains blue color pixel intensity values each 100 points for a year 

The files which are in below contain different infrared pixel intensity values each 100 points for a year **100points_pixelIntensitiesB9.csv, 100points_pixelIntensitiesB10.csv, 100points_pixelIntensitiesB10.csv**

**10k_random.csv** --> it contains coordinates of random 100 points in baltic region

**100points_iou.csv** --> it contains iou values of each 100 points for a year which were gotten using SAMGeo segmentation model

**100points_scores.csv** --> it contains score values of each 100 points for a year which were gotten using SAMGeo segmentation model (predict function)

**100points_probabilities.json** --> it contains segmentation class probabilities of center pixel of each image. 2nd github link was used for that

**VAR_results.json** --> it contains VAR Model result for each 100 points. Structure of the json is in below.

```
                     [
                        {
                            'rmse_results': {
                                'rmse_b4': rmse_result[0], 'rmse_b3': rmse_result[1], 'rmse_b2': rmse_result[2],
                                'rmse_iou': rmse_result[3], 'rmse_score': rmse_result[4],
                                'rmse_class1': rmse_result[5], 'rmse_class2': rmse_result[6], 'rmse_class3': rmse_result[7],
                                'rmse_class4': rmse_result[8], 'rmse_class5': rmse_result[9], 'rmse_class6': rmse_result[10]
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
                                'conf_mid_class6': conf_mid_class6.tolist(), 'conf_lower_class6': conf_lower_class6.tolist(), 'conf_upper_class6': conf_upper_class6.tolist()
                            }
                        }
                    ]
```

## Code Files

**VAR_Model.ipynb** --> Main noteebook for Variational Autoregression Analysis (VAR)

**cloud_probability.ipynb** --> Collect cloud probabilities for selected points and date interval

**pixel_intensities.ipynb** --> Collect pixel intentisty values for selected bands

**class_probabilities.ipynb** --> Collect class probabilities of segmented images

**create_df_different_points.ipynb** --> Collect random 10000 points beetween selected latitudes and longitudes


## First Experiment Results:

![diagram](https://github.com/kursatkomurcu/semantic_segmentation_for-_change_detection_in_satellite_imaging/blob/main/images/diagram.png)

This diagram shows the results of index 3

| Index | Lat      | Lon      | RMSE     | AIC      | Fall In CI |
|-------|----------|----------|----------|----------|------------|
| 0     | 57.9822  | 27.5759  | 0.085    | -92.993  | %0         |
| 1     | 54.8303  | 21.8945  | 0.118    | -123.880 | %0         |
| 2     | 59.1785  | 24.5851  | 0.058    | -92.618  | %0         |
| 3     | 57.2123  | 24.1739  | <0.001   | None     | %99.363    |
| 4     | 55.4973  | 23.1317  | 0.089    | -105.224 | %0         |
| 5     | 59.5876  | 25.7885  | 0.053    | -96.585  | %0.746     |
| 6     | 57.2948  | 22.5929  | 0.093    | -98.745  | %0         |
| 7     | 53.6124  | 27.2380  | 0.080    | -102.598 | %0         |
| 8     | 54.6356  | 22.8023  | 0.096    | -113.109 | %0         |
| 9     | 56.6370  | 20.7791  | 0.112    | -123.783 | %0         |

*Note: RMSE and Fall In CI columns are values for Class2*

## Full Results of Index 3

![pixel_intensities](https://github.com/kursatkomurcu/semantic_segmentation_for-_change_detection_in_satellite_imaging/blob/main/images/pixel_intensities.png)

![iou_scores](https://github.com/kursatkomurcu/semantic_segmentation_for-_change_detection_in_satellite_imaging/blob/main/images/iou_scores.png)

![classes](https://github.com/kursatkomurcu/semantic_segmentation_for-_change_detection_in_satellite_imaging/blob/main/images/classes.png)

![classes_](https://github.com/kursatkomurcu/semantic_segmentation_for-_change_detection_in_satellite_imaging/blob/main/images/classes_.png)

## How to test?

```bash
  pip install earthengine-api
```

```bash
  earthengine authenticate
```

```bash
  python test.py --start_date 2022-01-01 --end_date 2023-01-01 --bands B4,B3,B2 --index 3
```

## Citiation

```
  @article{komurcu2024semantic,
    author    = {K. K{\"o}m{\"u}rc{\"u} and L. Petkevicius},
    title     = {Semantic Segmentation for Change Detection in Satellite Imaging},
    journal   = {open-series},
    pages     = {57--64},
    month     = {May},
    year      = {2024},
    doi       = {10.15388/LMITT.2024.8}
  }
```
