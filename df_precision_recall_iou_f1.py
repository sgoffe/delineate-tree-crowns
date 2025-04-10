import deepforest
from deepforest import get_data
import pandas as pd 
import geopandas as gpd
from deepforest import IoU
from deepforest import evaluate
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt



# def convert_to_dictionaries(df):
#     """Converts a GeoDataFrame to a list of dictionaries."""
#     image_groups = df.groupby("image_path")
#     results = []
#     for image_path, group in image_groups:
#       boxes = []
#       labels = []
#       for index, row in group.iterrows():
#         boxes.append([float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])])
#         labels.append(0)

#       results.append({
#           "image_path": image_path,
#           "boxes": torch.tensor(boxes, dtype=torch.float32),
#           "labels": torch.tensor(labels, dtype=torch.int64)
#       })
#     results.sort(key=lambda x: x["image_path"])
#     return results



#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/mike_FL012_50m_sections_processed/annotations_mike_updated.csv')
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/sophia_FL012_50m_sections_processed/annotations_sophia_updated.csv')
#csv_file_predictions = get_data("/datalake/fdavis/Trees/data/mike_FL012_50m_sections_processed/annotations_mike_updated.csv")
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/julia_FL012_50m_sections_processed/annotations_julia_updated.csv')
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/matheson_FL012_50m_sections_processed/annotations_matheson_updated.csv')
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/luke_FL012_50m_sections_processed/annotations_luke_updated.csv')

# csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/matheson_FL017_50m_sections_processed/annotations_matheson_updated.csv')
csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations_mike_updated.csv')
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/annotations_sophia_updated.csv')
#csv_file_predictions = get_data("/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations_mike_updated.csv")
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/julia_FL017_50m_sections_processed/annotations_julia_updated.csv')
#csv_file_ground_truth = get_data('/datalake/fdavis/Trees/data/luke_FL017_50m_sections_processed/annotations_luke_updated.csv')

#csv_file_predictions = get_data('/datalake/fdavis/Trees/DeepForest/predictions_300_epoch.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/sophia_FL012_50m_sections_processed/annotations_sophia_updated.csv')
# csv_file_predictions = get_data('/datalake/fdavis/Trees/delineate-tree-crowns/yolo_predictions_updated.csv')
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_9_epoch.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_0_epoch.csv")
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/julia_FL012_50m_sections_processed/annotations_julia_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/matheson_FL012_50m_sections_processed/annotations_matheson_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/luke_FL012_50m_sections_processed/annotations_luke_updated.csv')

#csv_file_predictions = get_data("/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/annotations_sophia_updated.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_300_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_9_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/DeepForest/predictions_0_epoch_17.csv")
#csv_file_predictions = get_data("/datalake/fdavis/Trees/delineate-tree-crowns/yolo_predictions_17_updated.csv")
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/julia_FL017_50m_sections_processed/annotations_julia_updated.csv')
#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations_mike_updated.csv')

#csv_file_predictions = get_data('/datalake/fdavis/Trees/data/matheson_FL017_50m_sections_processed/annotations_matheson_updated.csv')
csv_file_predictions = get_data('/datalake/fdavis/Trees/data/luke_FL017_50m_sections_processed/annotations_luke_updated.csv')


ground_truth = pd.read_csv(csv_file_ground_truth)
predictions = pd.read_csv(csv_file_predictions)

print(ground_truth.dtypes)
print(predictions.dtypes)

ground_truth["geometry"] = ground_truth["geometry"].apply(wkt.loads)
predictions["geometry"] = predictions["geometry"].apply(wkt.loads)

ground_truth = gpd.GeoDataFrame(ground_truth, geometry="geometry")
predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

print(ground_truth.dtypes)
print(predictions.dtypes) 

result = evaluate.evaluate_boxes(predictions=predictions , ground_df=ground_truth, root_dir="/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/images", savedir="/datalake/fdavis/Trees/DeepForest/LW_ML")     
print(result)
print(result.keys())

results = result["results"]

result["match"] = results.IoU > 0.4
IoU = results.IoU.sum() / results.shape[0]
true_positive = sum(result["match"])
recall = true_positive / results.shape[0]
precision = true_positive / predictions.shape[0]
f1 = (2 * precision * recall) / (precision + recall)
print("precision: ", precision)
print("recall: ", recall)
print("F1: ", f1)
print("IoU: ", IoU )
