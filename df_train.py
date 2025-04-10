import os
import deepforest
import pandas as pd
from deepforest import main

from deepforest import get_data
from deepforest import visualize
from deepforest import evaluate


model = main.deepforest()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

#or
#main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
# model.use_release()

model.config['gpus'] = '-1'
model.config["train"]["csv_file"] = "/datalake/fdavis/Trees/data/1through5_and_augmented_data/train/annotations.csv"
model.config["train"]["root_dir"] = "/datalake/fdavis/Trees/data/1through5_and_augmented_data/train/images/"
model.config["validation"]["csv_file"] = "/datalake/fdavis/Trees/data/1through5_and_augmented_data/val/annotations.csv"
model.config["validation"]["root_dir"] = "/datalake/fdavis/Trees/data/1through5_and_augmented_data/val/images/"
# model.config["score_thresh"] = 0.4
model.config["train"]['epochs'] = 300
model.config["train"]["workers"] = 4



model.create_trainer()

model.trainer.fit(model)

model.trainer.save_checkpoint("/datalake/fdavis/Trees/DeepForest/checkpoint_300.pl")

csv_file = get_data("/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/annotations.csv")
predictions = model.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file).replace("sophia_FL017_50m_sections_processed","sophia_FL017_50m_sections_processed/images"))
predictions.head()
predictions.to_csv('predictions_300_epoch_17.csv', index=False, columns=["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"], na_rep=0)

results = model.evaluate(csv_file, root_dir="/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/images", iou_threshold = 0.4)

print("results of evaluate")
print(results["results"])
box_precision = results["box_precision"]
print(f'box precision: {box_precision}')
box_recall = results["box_recall"]
print(f"box recall: {box_recall}")

ground_truth = pd.read_csv(csv_file)
ground_truth.head()

visualize.plot_prediction_dataframe(predictions, root_dir = "/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/images/", savedir="/datalake/fdavis/Trees/DeepForest/eval_df_after_train_output/300_17", ground_truth=ground_truth)