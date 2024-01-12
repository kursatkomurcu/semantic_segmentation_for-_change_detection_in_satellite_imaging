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

```json
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
