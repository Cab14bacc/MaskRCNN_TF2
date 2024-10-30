import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import mrcnn.utils
import cv2
import os
import argparse
import skimage
import numpy as np
import json


# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

# dataset_dir
# |___labels.txt
# |___annot.json (all annotations)
# |
# |___logs (dir to store model logs)
# |   |__
# |___output_weights (where the weights will be stored)
# |   |__
# |___train_images
# |   |
# |   |____0011.jpg
# |   |____0052.jpg
# |
# |___val_images
# |   |
# |   |____0021.jpg
# |   |____0032.jpg  




parser = argparse.ArgumentParser(description="custom object detection")
parser.add_argument("-f", "--file", required=True, help="target image")
parser.add_argument("-a", "--annot", required=True, help="annotation json")
parser.add_argument("-l", "--labels", required=True, help="labels.txt")
parser.add_argument("-w", "--weight", required=True, help="weight used for prediction")

args = parser.parse_args()

CLASS_NAMES = ['BG']
MODEL_LOGS_DIR = str()


with open(os.path.normpath(args.labels), 'r') as f:
    CLASS_NAMES.extend(f.read().split('\n'))

print("label names: ", CLASS_NAMES)

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=MODEL_LOGS_DIR)

# Load the weights into the model.
model.load_weights(filepath=os.path.normpath(args.weight), 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imdecode(np.fromfile(os.path.normpath(args.file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image, 
#                                   boxes=r['rois'], 
#                                   masks=r['masks'], 
#                                   class_ids=r['class_ids'], 
#                                   class_names=CLASS_NAMES, 
#                                   scores=r['scores'])



# image_id = 0
# while image_id < 50:
#     # define image id 
#     image_id += 1
#     # load the image
#     image = train_dataset.load_image(image_id)
#     # load the masks and the class ids
#     mask, class_ids = train_dataset.load_mask(image_id)
#     # extract bounding boxes from the masks
#     bbox = mrcnn.utils.extract_bboxes(mask)
#     # display image with masks and bounding boxes
#     display_instances(image, bbox, mask, class_ids, train_dataset.class_names)

annot_path = os.path.normpath(args.annot)
image_path= os.path.normpath(args.file)

def load_image_data(img_path, annot_path):
    
    img_dir = ""
    annotation = {}
    with open(annot_path) as f: 
        annotation = json.load(f)

    img_name = os.path.basename(os.path.normpath(img_path))
    cur_annot = annotation[img_name]

    # each dict ("x" : ..., "y" : ...) represent a polygon, "all points x" is a list of x coordinates, 
    polygons = [{"x" : item["shape_attributes"]["all_points_x"], "y" : item["shape_attributes"]["all_points_y"], "label" : item["region_attributes"]["label"]} for item in cur_annot["regions"].values()]
    img = skimage.io.imread(img_path)
    # how many rows
    height = img.shape[0]
    width = img.shape[1]

    masks = np.zeros([height, width, len(polygons)], dtype='uint8')

    class_ids = []
    for i in range(len(polygons)):
        rr, cc = skimage.draw.polygon(polygons[i]["y"] ,polygons[i]["x"])
        masks[rr,cc, i] = 1
        class_ids.append(CLASS_NAMES.index(polygons[i]["label"]))
    b_boxes = mrcnn.utils.extract_bboxes(masks)

    # return: 
    #   masks: masks in the shape [height, width, instances] 
    #   class_ids: a 1D array of class IDs of the instance masks.\
    #   b_boxes: bounding boxes of the masks
    return masks, np.asarray(class_ids, dtype='int32'), b_boxes


gt_masks, gt_class_ids, gt_b_boxes = load_image_data(image_path, annot_path)


# def display_differences(image,
#                         gt_box, gt_class_id, gt_mask,
#                         pred_box, pred_class_id, pred_score, pred_mask,
#                         class_names, title="", ax=None,
#                         show_mask=True, show_box=True,
#                         iou_threshold=0.5, score_threshold=0.5):

# Visualize the detected objects.
mrcnn.visualize.display_differences(image=image, 
                                    gt_mask=gt_masks,
                                    gt_box = gt_b_boxes,
                                    gt_class_id=gt_class_ids,
                                  pred_box=r['rois'], 
                                  pred_mask=r['masks'], 
                                  pred_class_id=r['class_ids'], 
                                  pred_score=r['scores'],
                                  class_names=CLASS_NAMES )


