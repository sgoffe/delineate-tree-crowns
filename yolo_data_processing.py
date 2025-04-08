# -*- coding: utf-8 -*-

# !pip install numpy
# !pip install matplotlib
# !pip install imageio

import os
import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import skimage as ski
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# # **To Be Set**
forest_file = iio.imread(uri="/datalake/fdavis/Trees/data/50m_sections_unprocessed/mike_FL017/RU_CYN_TR2_FL017_train50a.tif")
labels = gpd.read_file("/datalake/fdavis/Trees/data/50m_sections_unprocessed/mike_FL017/RU_CYN_TR2_FL017_train50a_crowns_ML_bboxes.shx")
min_x, min_y, max_x, max_y = 2140773.1684259800240397,2497924.7962134201079607,2140834.5699665797874331,2497869.0271338201127946
# FL012 2161398.7699114200659096,2439639.8628751100040972, 2161460.1782242199406028, 2439584.1068095099180937
# comes in format 1, 2, 3, 4 but format to enter it into script is 1, 4, 3, 2


tile_dimensions = 768

subset = "RU_CYN_TR2_FL017_train50a_crowns_tiles"
file_path = "/datalake/fdavis/Trees/data/mike_FL017_50m_sections_processed/"
# # **To Be Set**

def normalize(forest):
  return (forest - forest.min()) / (forest.max() - forest.min())

forest = np.array(forest_file)
forest_normalized = normalize(forest)

image_height, image_width  = forest.shape[:2]
print(image_height)
print(image_width)

def geographic_coordinate_to_pixel(x, y, min_x, min_y, max_x, max_y, image_width, image_height):
    px = int((x - min_x) / (max_x - min_x) * image_width)
    py = int((y - min_y) / (max_y - min_y) * image_height)
    return (px,py)

def polygon_to_pixel(polygon, min_x, min_y, max_x, max_y, image_width, image_height):
    if isinstance(polygon, Polygon):  # Single polygon
        pixel_coords = [geographic_coordinate_to_pixel(x, y, min_x, min_y, max_x, max_y, image_width, image_height)
                        for x, y in polygon.exterior.coords]
        return Polygon(pixel_coords)
    elif isinstance(polygon, MultiPolygon):  # Multi-polygon
        pixel_polygons = [polygon_to_pixel(p, min_x, min_y, max_x, max_y, image_width, image_height)
                          for p in polygon.geoms]
        return MultiPolygon(pixel_polygons) # Return as a MultiPolygon
    else:
        raise ValueError(f"Unsupported geometry type: {type(polygon)}")

"""convert label coordinates from geographic to pixel"""

pixel_polygons = labels['geometry'].apply(lambda poly: polygon_to_pixel(poly, min_x, min_y, max_x, max_y, image_width, image_height))

def polygon_to_bbox_format(polygon):
    # Get the minimum and maximum x and y values
    x_min, y_min, x_max, y_max = polygon.bounds
    return x_min, y_min, x_max, y_max

def bbox_to_center_format(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return center_x, center_y, width, height

"""convert polygons to bbox format and then to center format bboxes"""

bounding_boxes = pixel_polygons.apply(polygon_to_bbox_format)
center_format_bboxes = bounding_boxes.apply(bbox_to_center_format)

"""plot image and labels together"""

fig, ax = plt.subplots()
ax.imshow(forest_normalized, cmap='gray')

for bbox in center_format_bboxes:
    center_x, center_y, width, height = bbox
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
plt.savefig("image_with_labels_mike_17.png")

"""pad image to allow for even sectioning"""

offset_y = tile_dimensions - forest.shape[1] % tile_dimensions
offset_x = tile_dimensions - forest.shape[0]  % tile_dimensions

forest_padded = np.pad(forest_normalized, ((0,offset_x),(offset_y, 0),(0,0)), 'constant', constant_values=0)

y = forest_padded.shape[0]
x = forest_padded.shape[1]

def pad_bbox(bbox):
  x_center, y_center, width, height = bbox
  return x_center + offset_y, y_center, width, height

padded_center_format_bboxes = center_format_bboxes.apply(pad_bbox)

"""plot padded image and labels together"""

fig, ax = plt.subplots()
ax.imshow(forest_padded, cmap='gray')

for bbox in padded_center_format_bboxes:
    center_x, center_y, width, height = bbox
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
plt.savefig("padded_image_with_labels_mike_17.png")

def make_tiles(image, labels):
  tiles = []
  annotations = []
  num_tiles_x = x // tile_dimensions
  num_tiles_y = y // tile_dimensions
  
  os.makedirs(os.path.dirname(f"{file_path}images/"), exist_ok=True)

  for i in range(num_tiles_x):
      for j in range(num_tiles_y):
          tile_x = i * tile_dimensions
          tile_y = j * tile_dimensions
          tile = image[tile_y:tile_y + tile_dimensions, tile_x:tile_x + tile_dimensions, :]

          tile_bboxes = []

          # Loop through each bounding box and check if it intersects the current tile
          for x_center, y_center, width, height in labels:
              x_min = x_center - width / 2
              y_min = y_center - height / 2
              x_max = x_center + width / 2
              y_max = y_center + height / 2

              if not (x_max < tile_x or x_min > tile_x + tile_dimensions or y_max < tile_y or y_min > tile_y + tile_dimensions):
                  adj_x_min = max(0, x_min - tile_x)  # Adjust x_min relative to the tile's x-coordinate
                  adj_y_min = max(0, y_min - tile_y)  # Adjust y_min relative to the tile's y-coordinate
                  adj_x_max = min(tile_dimensions, x_max - tile_x)  # Adjust x_max relative to the tile's x-coordinate
                  adj_y_max = min(tile_dimensions, y_max - tile_y)  # Adjust y_max relative to the tile's y-coordinate

                  # Only add bounding boxes with positive width and height (not empty boxes)
                  if adj_x_max > adj_x_min and adj_y_max > adj_y_min:
                      # Recalculate the center and dimensions of the bounding box relative to the tile
                      x_center_adj = (adj_x_min + adj_x_max) / 2
                      y_center_adj = (adj_y_min + adj_y_max) / 2
                      width_adj = adj_x_max - adj_x_min
                      height_adj = adj_y_max - adj_y_min

                      #normalize
                      x_center_adj = x_center_adj / tile_dimensions
                      y_center_adj = y_center_adj / tile_dimensions
                      width_adj = width_adj / tile_dimensions
                      height_adj = height_adj / tile_dimensions

                      tile_bboxes.append((x_center_adj, y_center_adj, width_adj, height_adj))

          annotations.append((tile, tile_bboxes))

          current_path = f"{file_path}labels/forest_{subset}_{i}_{j}.txt"
          os.makedirs(os.path.dirname(current_path), exist_ok=True)

          with open(f"{file_path}labels/forest_{subset}_{i}_{j}.txt", 'a') as f:
            for x_center, y_center, width, height in tile_bboxes:
              f.write(f"0 {x_center} {y_center} {width} {height}\n")

          tile = (tile * 255).astype(np.uint8)
          iio.imwrite(uri=(f"{file_path}images/forest_{subset}_{i}_{j}.tiff"), image=tile) #, extension= ".tiff")

make_tiles(forest_padded, padded_center_format_bboxes)

