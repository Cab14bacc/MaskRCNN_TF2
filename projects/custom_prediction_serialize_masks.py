import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import argparse
from datetime import datetime
import json

# Argument parser
parser = argparse.ArgumentParser(description="Custom object detection and serialization")
parser.add_argument("-s", "--source", required=True, help="Project Folder")
parser.add_argument("-w", "--weight", required=True, help="Weight used for prediction")
parser.add_argument("-d", "--images-directory", required=True, help="directory of images")
parser.add_argument("-o", "--output-file", default=None, help="prediction results in json format")

args = parser.parse_args()

# Load class names
CLASS_NAMES = ['BG']
DATASET_DIR = args.source
MODEL_LOGS_DIR = os.path.join(DATASET_DIR, "logs")
IMGS_DIR = os.path.normpath(args.images_directory)
PRED_RESULT_FILE = os.path.join(DATASET_DIR, "pred_output")

if(args.output_file is None):
    # project name + current time as default output filename
    PRED_RESULT_FILE = os.path.join(DATASET_DIR, "pred_output", os.path.basename(os.path.normpath(DATASET_DIR)) + "_PRED_" +  datetime.now().strftime("%Y%m%d%H%M") + ".json") 
else:
    PRED_RESULT_FILE = os.path.normpath(args.output_file) 

print("Project folder: ", DATASET_DIR)

with open(os.path.join(DATASET_DIR, "labels.txt"), 'r') as f:
    CLASS_NAMES.extend(f.read().strip().split('\n'))

print("Label names: ", CLASS_NAMES)


class CustomConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


# Load model
model = mrcnn.model.MaskRCNN(mode="inference", model_dir=MODEL_LOGS_DIR, config=CustomConfig())
model.load_weights(args.weight, by_name=True)

# Run object detection on images
output = {}
for image_name in os.listdir(IMGS_DIR):
    image_path = os.path.join(IMGS_DIR, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred_result = model.detect([image])[0]

    # detect returns:
    # rois: [N, (y1, x1, y2, xc2)] detection bounding boxes
    # class_ids: [N] int class IDs
    # scores: [N] float probability scores for the class IDs
    # masks: [H, W, N] instance binary masks

    output[image_name] = {}
    for i in range(len(pred_result["class_ids"])):
        # bbox = [y1, x1, y2, x2]
        bbox = pred_result["rois"][i]
        mask = pred_result["masks"][..., i]
        mask = mask[bbox[0]:(bbox[2] - 1),bbox[1]:(bbox[3] - 1)]
        output[image_name][str(i)] = {
            "Mask": mask.astype(int).tolist(),
            "bbox": bbox.tolist(),
            "label": CLASS_NAMES[pred_result["class_ids"][i]]
        }


# Serialize results to a file
with open(PRED_RESULT_FILE, 'w') as f:
    json.dump(output, f, separators=(",",":"))

print(f"Results saved to {PRED_RESULT_FILE}")