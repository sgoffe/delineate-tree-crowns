# delineate-tree-crowns

**TreeDetectionPipeline.py** — Python script to train and evaluate YOLOv8 model (Redmon et al., 2018)

To run: 

- set path to data.yaml file l20

- put number of images to display as argument to display_train() (default = 9) l70
, result saved to train_images.png
  
- put number of images to display as argument to predict_on_test() (default = 9) l138
, result saved to predictions.png
  
- set number of epochs l79



**DataProcessing.py** — Python script to process and tile image and annotations (image as tiff and annotations as shp/shx)

To run: 

- set path to tiff forest image l18 and shx annotations l19

- set coordinate bounds l20
-   can be found in QGIS in the forest tiff properties (comes in format 1, 2, 3, 4 but format to enter it into script is 1, 4, 3, 2
  
- set desired tile dimensions l22
  
- set file path l23 for images and labels folders to be stored

other things to set — path to image with labels l88 and path to padded image with labels l119


Sources:

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779–788. https://doi.org/10.1109/cvpr.2016.91



note: 
- will need to have installed matplotlib, imageio, numpy
- I think you have to have annotations.shp as well and in same directory as annotations.shx
