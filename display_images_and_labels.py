import cv2
import yaml
import os
import random
import matplotlib.pyplot as plt


train_images = '/datalake/fdavis/Trees/data/matheson_FL012_50m_sections_processed/images'
train_labels = '/datalake/fdavis/Trees/data/matheson_FL012_50m_sections_processed/labels'

def display_train(num_images = 9):
    image_files = os.listdir(train_images)
    random_images = random.sample(image_files, num_images)
    fig, axs = plt.subplots(num_images, 2, figsize=(24, 108))

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
    plt.savefig('train_images_matheson.png')

display_train(9)