#!/usr/bin/env python
# coding: utf-8

# # **Installing Dependencies**
import locale
import cv2
import yaml
import os
import random
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# # **Setting Paths to Data**
path = "/datalake/fdavis/Trees/data.yaml"

with open(path, 'r') as stream:
    paths_loaded = yaml.safe_load(stream)

train_images = paths_loaded.get('train')
train_labels = train_images.replace("images", "labels")

test_images = paths_loaded.get('test')
test_labels = test_images.replace("images", "labels")

val_images = paths_loaded.get('val')
val_labels = val_images.replace("images", "labels")

# # **Setting Configuration from YAML File**
with open('/datalake/fdavis/Trees/configuration.yaml', 'r') as stream:
    configuration = yaml.safe_load(stream)

configuration["data"] = path 
##alreadys set in configuration.yaml but maybe would be better to set here?


# # **Method to Display Train Images With Labels**
def display_train(num_images = 9):
    image_files = os.listdir(train_images)
    random_images = random.sample(image_files, num_images)
    fig, axs = plt.subplots(num_images, 2, figsize=(8, 36))

    for i, image_file in enumerate(random_images):
        image_path = os.path.join(train_images, image_file)
        image = cv2.imread(image_path)
    
        axs[i, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[i, 1].axis('off')
    
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(train_labels, label_file)
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
    
    
        axs[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[i, 0].axis('off')
    
    plt.show()
    plt.savefig('train_images.png')

display_train()


# # **Load pretrained model**
model = YOLO('yolov8x.pt')

#can also add 'weights': 'yolov8x.pt' to configuration.yaml

# #** Train Model**
model.train(**configuration)
model.save

# # **Evaluate model performance**
metrics = model.val(split = 'test',
                    project = 'test_directory',
                    name = 'test_val_experiment_name_dict'
                   )

# # **Make Predictions on Test Images**
def tree_detect(img_path):

    img = cv2.imread(img_path)
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img


def predict_on_test(num_images = 9):
    custom_image_dir = test_images
    image_files = os.listdir(custom_image_dir)
    selected_images = random.sample(image_files, num_images)
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 36))
    
    for i, img_file in enumerate(selected_images):
        img_path = os.path.join(custom_image_dir, img_file)
        detect_img = tree_detect(img_path)
        image = cv2.imread(img_path)
    
        axes[i, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 1].axis('off')
        
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(test_labels, label_file)
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
    
    
    
