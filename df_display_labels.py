##display labels
import deepforest
import os
import imageio
import imageio.v3 as iio
from deepforest import get_data
import pandas as pd 
import numpy as np
import geopandas as gpd
from deepforest import IoU
from deepforest import evaluate
from deepforest import visualize
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt


csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations_mike_updated.csv')

csv_file_predictions = get_data("/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/annotations_sophia_updated.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_300_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_9_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_0_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/delineate-tree-crowns/yolo_predictions_17_updated.csv")
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/julia_FL017_50m_sections_processed/annotations_julia_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations_mike_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/matheson_FL017_50m_sections_processed/annotations_matheson_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/luke_FL017_50m_sections_processed/annotations_luke_updated.csv')


forest_file =iio.imread(uri="/datalake/fdavis/Trees/data/julia_FL017_50m_sections_processed/images/forest_RU_CYN_TR2_FL017_train50a_crowns_tiles_1_2.tiff")

ground_truth = pd.read_csv(csv_file_ground_truth, skiprows=lambda x:x != 0 and "forest_RU_CYN_TR2_FL017_train50a_crowns_tiles_1_2.tiff" not in open(csv_file_ground_truth).readlines()[x])
predictions = pd.read_csv(csv_file_predictions, skiprows=lambda x:x != 0 and "forest_RU_CYN_TR2_FL017_train50a_crowns_tiles_1_2.tiff" not in open(csv_file_predictions).readlines()[x])

forest=np.array(forest_file)


ground_truth["geometry"] = ground_truth["geometry"].apply(wkt.loads)
predictions["geometry"] = predictions["geometry"].apply(wkt.loads)

ground_truth = gpd.GeoDataFrame(ground_truth, geometry="geometry")
predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

print(ground_truth.dtypes)
print(ground_truth)
print(predictions.dtypes)
print(predictions)

img = visualize.plot_results(results=ground_truth, ground_truth=predictions, results_color=[0,0,255], image=forest, ground_truth_color=[0,245,255],height=768, width=768,thickness=2, savedir="/datalake/fdavis/Trees", basename="SG_ML")
