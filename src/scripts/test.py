import ee
import argparse
from create_df_different_points import *
from pixel_intensities import *
from cloud_probability import *
from Automatically_generating_object_masks_with_SAM import *
from VAR_results_for_each_point import *
from VAR_Model import *

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=str, help='Format: yyyy-mm-dd')
parser.add_argument('--stop_date', type=str, help='Format: yyyy-mm-dd')
parser.add_argument('--bands', type=str, help='Separete by comma. Exp: B4,B3,B2')
parser.add_argument('--index', type=int, help='Row for test after collecting data')

args = parser.parse_args()
bands = args.bands.split(',')

ee.Initialize()

create_dataset_with_random_coordinates()
collect_pixel_intensities(args.start_date, args.stop_date, bands)
collect_cloud_probabilities(args.start_date, args.end_date)
collect_iou(args.start_date, args.end_date)
collect_scores(args.start_date, args.end_date)
VAR_Results_for_each_point()
VAR_Model(args.index)
