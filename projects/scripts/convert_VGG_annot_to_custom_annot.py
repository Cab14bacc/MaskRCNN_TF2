import argparse 
import json
import os
import numpy as np
import skimage 
import mrcnn.utils as utils

# json format: 
# {
#     "圖檔名": {
#       "0": {
#         "Mask": [[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                  [1, 1, 1, 0, 1, 0, 1, 1, 0],
#                  [1, 1, 1, 0, 1, 0, 1, 1, 0]],
#         "bbox": [x1, y1, x2, y2], //矩形範圍，基本上就是從Mask算出來的
#         "label": "label名稱 (e.g. CrossWalk)"
#       },
#       .
#       .
#       .
#       "49": {
#         "Mask": [[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                  [1, 1, 1, 0, 1, 0, 1, 1, 0],
#                  [1, 1, 1, 0, 1, 0, 1, 1, 0]],
#         "bbox":[x1, y1, x2, y2],
#         "label": "label名稱 (e.g. CrossWalk)"
#       }
#     }
#     .
#     .
#     "0050.png": {
#         "0": {
#           "Mask": [[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [1, 1, 1, 0, 1, 0, 1, 1, 0],
#                    [1, 1, 1, 0, 1, 0, 1, 1, 0]],
#           "bbox":[x1, y1, x2, y2],
#           "label": "label名稱 (e.g. CrossWalk)"
#         },
#         .
#         .
#         .
#         "49": {
#           "Mask": [[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [1, 1, 1, 0, 1, 0, 1, 1, 0],
#                    [1, 1, 1, 0, 1, 0, 1, 1, 0]],
#           "bbox":[x1, y1, x2, y2],
#           "label": "label名稱 (e.g. CrossWalk)"
#         }
#       }
#   }

parser = argparse.ArgumentParser(description="Take in a VGG annotation json, and convert them into a custom JSON file")
parser.add_argument("-o", "--output", default="annot.json", help="output json filepath and name")
parser.add_argument("-s", "--source-file", required=True, help="source VGG annot json")
parser.add_argument("-d", "--source-dir", required=True, help="source images directory that corresponds to the annotations")

args = parser.parse_args()

# source json format 
# "0050.png": {
#     "fileref": "",
#     "size": 226354,
#     "filename": "0050.png",
#     "base64_img_data": "",
#     "file_attributes": {},
#     "regions": {
#       "0": {
#         "shape_attributes": {
#           "name": "polygon",
#           "all_points_x": [
#             419.29595827900914, 281.61668839634945, 282.4511082138201,
#             430.97783572359845, 419.29595827900914
#           ],
#           "all_points_y": [
#             347.9530638852673, 275.35853976531945, 247.82268578878748,
#             324.5893089960887, 347.9530638852673
#           ]
#         },
#         "region_attributes": { "label": "CrossWalk" }
#       },
#       "1": {
#         "shape_attributes": {
#           "name": "polygon",
#           "all_points_x": [
#             434.3155149934811, 436.8187744458931, 461.0169491525424,
#             460.1825293350717, 434.3155149934811
#           ],
#           "all_points_y": [
#             360.4693611473273, 437.23598435462844, 437.23598435462844,
#             359.6349413298566, 360.4693611473273
#           ]
#         },
#         "region_attributes": { "label": "CrossWalk" }
#       },
#       "2": { ...

# source json
source = {}
result = {}

IMG_DIR = os.path.normpath(args.source_dir)

# load in src json
with open(os.path.join(args.source_file)) as f:
    source = json.load(f)
    for image_data in source.items():
        image_path = os.path.join(IMG_DIR, image_data[0])
        img = skimage.io.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]
        polygons = [{"x" : mask["shape_attributes"]["all_points_x"], "y" : mask["shape_attributes"]["all_points_y"], "label" : mask["region_attributes"]["label"]} for mask in image_data[1]["regions"].values()]
        masks = np.zeros((height, width, len(polygons)))
        labels = [poly["label"] for poly in polygons]
        for i in range(len(polygons)):
            rr, cc = skimage.draw.polygon(polygons[i]["y"] ,polygons[i]["x"])
            masks[rr,cc, i] = 1
        b_boxes = utils.extract_bboxes(masks)

        # image_basename_no_extension = image_data[0].split('.')[0]
        # result[image_basename_no_extension] = {}
        result[image_data[0]] = {}
        for i in range(len(polygons)):
            # bbox = [y1, x1, y2, x2]
            bbox = b_boxes[i]
            # switch to output x1 y1 x2 y2
            mask = masks[..., i]
            mask = mask[bbox[0]:(bbox[2] - 1),bbox[1]:(bbox[3] - 1)]
            bbox = bbox[[1,0,3,2]]
            result[image_data[0]][str(i)] = {
                "Mask": mask.astype(int).tolist(),
                "bbox": bbox.tolist(),
                "label": labels[i]
        }



# output result to result json
with open(os.path.normpath(args.output), "w+") as outputFile:
    json.dump(result, outputFile, separators=(",",":"))

print(f"Results saved to {os.path.normpath(args.output)}")