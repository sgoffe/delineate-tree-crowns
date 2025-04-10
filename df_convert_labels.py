import os
import pandas as pd
from PIL import Image

def yolo_to_csv(yolo_dir, output_csv):
    rows = []

    for file in os.listdir(yolo_dir):
        if file.endswith(".txt"):
            img_file = file.replace(".txt", ".tiff")  # Adjust for your image format
            img_path = os.path.join(yolo_dir, img_file).replace("labels", "images")
            label_path = os.path.join(yolo_dir, file)

            # Check if the image exists
            if not os.path.exists(img_path):
                print(f"Image {img_file} not found, skipping...")
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) != 5:
                        print(f"Skipping malformed line in {file}: {line}")
                        continue

                    class_id, x_center, y_center, w, h = map(float, values)
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    x_max = (x_center + w / 2) * width
                    y_max = (y_center + h / 2) * height

                    rows.append([img_file, x_min, y_min, x_max, y_max, "Tree"])

    df = pd.DataFrame(rows, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

yolo_to_csv("/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/labels", "/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/annotations.csv")

#yolo_dir (str): Path to the directory containing YOLO annotation files and images.
#output_csv (str): Path to save the output CSV file.