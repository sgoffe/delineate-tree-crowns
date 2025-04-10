import os
import deepforest
import pandas as pd
from deepforest import main
from deepforest import get_data
from deepforest import visualize
from deepforest import evaluate

# m = main.deepforest()
# m.load_model(model_name="weecology/deepforest-tree", revision="main")
m = main.deepforest.load_from_checkpoint("/datalake/fdavis/Trees/DeepForest/checkpoint_9.pl")

csv_file = get_data("/datalake/fdavis/Trees/data/sophia_FL012_50m_sections_processed/annotations.csv")
predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file).replace("sophia_FL012_50m_sections_processed","sophia_FL012_50m_sections_processed/images"))
predictions.head()
predictions.to_csv('predictions_9_epoch_12_hm.csv', index=False, columns=["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"], na_rep=0)

ground_truth = pd.read_csv(csv_file)
ground_truth.head()

visualize.plot_prediction_dataframe(predictions, root_dir = '/datalake/fdavis/Trees/data/sophia_FL012_50m_sections_processed/images/', savedir="/datalake/fdavis/Trees/DeepForest/evaluate_df_output4", ground_truth=ground_truth)

# metrics = evaluate.evaluate(ground_truth=ground_truth, predictions=predictions) this func doesnt seem to work/exist...?

results = m.evaluate(csv_file, root_dir=os.path.dirname(csv_file).replace("sophia_FL012_50m_sections_processed","sophia_FL012_50m_sections_processed/images"), savedir="/datalake/fdavis/Trees/DeepForest/evaluate_df_output_9_epoch_12_hm")

print("results of evaluate")
print(results["results"])
box_precision = results["box_precision"]
print(f'box precision: {box_precision}')
box_recall = results["box_recall"]
print(f"box recall: {box_recall}")

iou = sum(results["results"].IoU) / results["results"].shape[0]
print(f"IoU: {iou}")