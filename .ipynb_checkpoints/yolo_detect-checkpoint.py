# # **Installing Dependencies**
import locale
import cv2
import csv
import yaml
import os
import random
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


source = "/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/images/"

model = YOLO('/datalake/fdavis/Trees/test_directory/yolo_3002/weights/best.pt')

results = model(source, stream=True)

data = [
    ['image_path','xmin','ymin','xmax','ymax','label']
]

for result in results:
    csv_result = []
    csv_result.append(result.path)
    boxes = result.boxes.xyxy
    for box in boxes:
        data.append([os.path.basename(result.path)] + box.tolist() + [0])


with open('yolo_predictions_17.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)


## **visualize results**
def tree_detect(img_path):

    img = cv2.imread(img_path)
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img


def predict_on_test(num_images = 9):
    custom_image_dir = '/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/images'
    image_files = os.listdir(custom_image_dir)
    selected_images = random.sample(image_files, num_images)
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 36))
    
    for i, img_file in enumerate(selected_images):
        img_path = os.path.join(custom_image_dir, img_file)

        print('\n\n' + img_path +'\n')
        
        detect_img = tree_detect(img_path)
        image = cv2.imread(img_path)
    
        axes[i, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 1].axis('off')
        
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join('/datalake/fdavis/Trees/data/sophia_FL017_50m_sections_processed/labels', label_file)
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
    
        for label in labels:
            if len(label.split()) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, label.replace(',', '').split())
            x_min = int((x_center - width/2) * image.shape[1])
            y_min = int((y_center - height/2) * image.shape[0])
            x_max = int((x_center + width/2) * image.shape[1])
            y_max = int((y_center + height/2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
        axes[i, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 1].axis('off')
        
        axes[i, 0].imshow(detect_img)
        axes[i, 0].axis('off')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('predictions.png')

predict_on_test()