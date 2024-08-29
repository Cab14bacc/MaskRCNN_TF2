import mrcnn.utils as utils
import numpy as np
import os
import argparse

import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import skimage
from datetime import datetime
import csv
import json

# Argument parser
parser = argparse.ArgumentParser(description="Custom object detection and serialization")
# parser.add_argument("-s", "--source", required=True, help="Project Folder")
parser.add_argument("-l", "--labels", required=True, help="location of the list of labels")
parser.add_argument("-a", "--annot", required=True, help="location of the VGG annotations")
parser.add_argument("-w", "--weight", required=True, help="Weight used for prediction")
parser.add_argument("-d", "--images-directory", required=True, help="directory of images")
parser.add_argument("-o", "--output", default="./result.csv", help="location of the csv file where the result will be stored")



args = parser.parse_args()

# Load class names
CLASS_NAMES = ['BG']
IMGS_DIR = os.path.normpath(args.images_directory)
# PRED_RESULT_FILE = os.path.normpath(args.output_file)


with open(os.path.normpath(args.labels), 'r') as f:
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



def load_image_data(img_path, annot_path):
    
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

# returns gt pred pairs where the iou is above a certain threshold, used for later analysis
def compute_gt_pred_pairs(class_names , gt_boxes, gt_class_ids, gt_masks,
                        pred_boxes, pred_class_ids, pred_scores, pred_masks,
                        iou_threshold=0.3, score_threshold=0.0):
    
    """Finds matches between prediction and ground truth instances.

    Returns:
        result: a array of rows of gt pred pairs (gt, pred, pred score, IfMatched) 
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream

    # gt_boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    gt_boxes = utils.trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    gt_class_ids = gt_class_ids[:gt_boxes.shape[0]]

    pred_boxes = utils.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    pred_class_ids = pred_class_ids[:pred_boxes.shape[0]]
 
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    # overlaps[i, j] will return the IoU between the ith pred_mask and the jth gt_mask
    overlaps = utils.compute_overlaps_masks(pred_masks, gt_masks)

    # edge cases where no overlaps, either pred and no overlapping gt, or gt and no overlapping pred
    pred_ixs_no_overlap_gt = np.where(np.all(overlaps < iou_threshold, axis=1))[0]
    gt_ixs_no_overlap_pred = np.where(np.all(overlaps < iou_threshold, axis=0))[0]

    result = []

    # the ending 0 and 1 indicate matched(1) pairs or not(0)

    for idx in pred_ixs_no_overlap_gt:
        # print(f" {'None':<20} || { class_names[pred_class_ids[idx]]:<20} || {'None':<20} || 0")
        result.append(["None", class_names[pred_class_ids[idx]], "None", "0"])

    for idx in gt_ixs_no_overlap_pred:
        # print(f" {class_names[gt_class_ids[idx]]:<20} || {'None':<20}|| {'None':<20} || 0")
        result.append([class_names[gt_class_ids[idx]], "None", "None", "0"])

    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low iou
        low_iou_idx = np.where(overlaps[i, sorted_ixs] < iou_threshold)[0]
        if low_iou_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_iou_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            if  class_names[pred_class_ids[i]] ==  class_names[gt_class_ids[j]]:
                # print(f" {class_names[gt_class_ids[j]]:<20} || { class_names[pred_class_ids[i]]:<20} || {pred_scores[i]:<20.3f}" || "1")
                result.append([class_names[gt_class_ids[j]], class_names[pred_class_ids[i]], pred_scores[i], "1"])
            else:
                # print(f" {class_names[gt_class_ids[j]]:<20} || { class_names[pred_class_ids[i]]:<20} || {pred_scores[i]:<20.3f} || 0")
                result.append([class_names[gt_class_ids[j]], class_names[pred_class_ids[i]], pred_scores[i], "0"])

    return result









# Load model
model = mrcnn.model.MaskRCNN(mode="inference", model_dir=str(), config=CustomConfig())
print("\n\nweight path: ",os.path.normpath(args.weight), "\n\n")
model.load_weights(filepath=os.path.normpath(args.weight), by_name=True)
annot_path = os.path.normpath(args.annot)




fields = ["Img", "GT", "Pred", "Score", "IfValid"]
filename = os.path.normpath(args.output)
# for each image output results to csv

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for image_name in os.listdir(IMGS_DIR):

        print("Image: ", image_name)
        image_path = os.path.join(IMGS_DIR, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load in prediction results 
        pred_result = model.detect([image])[0]

        # load in gt annotations
        gt_masks, gt_class_ids, gt_bboxes = load_image_data(image_path, annot_path)

        # log gt prediction pairs 
        result = compute_gt_pred_pairs(  gt_masks = gt_masks,
                                gt_boxes = gt_bboxes,
                                gt_class_ids = gt_class_ids,
                                pred_boxes = pred_result['rois'], 
                                pred_masks = pred_result['masks'], 
                                pred_class_ids = pred_result['class_ids'], 
                                pred_scores = pred_result['scores'],
                                class_names = CLASS_NAMES,
                                iou_threshold = 0.3 )
        result = np.array(result)
        image_name_column = np.full((result.shape[0], 1), image_name)
        result = np.concatenate((image_name_column, result), axis=1)
        csvwriter.writerows(result)
        print(result)
        print("\n\n\n")