import mrcnn.utils as utils
import numpy as np
import os
import argparse
import sys
import mrcnn.config
import mrcnn.model
import mrcnn.visualize as visualize
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours

import cv2
import skimage
from datetime import datetime
import csv
import json
import re

# Argument parser
parser = argparse.ArgumentParser(description="Custom object detection and serialization")
# parser.add_argument("-s", "--source", required=True, help="Project Folder")
parser.add_argument("-l", "--labels", required=True, help="location of the list of labels")
parser.add_argument("-a", "--annot", required=True, help="location of the VGG annotations")
parser.add_argument("-w", "--weight", required=True, help="Weight used for prediction")
parser.add_argument("-d", "--images-directory", required=True, help="parent directory of images, combine with SUBFOLDERS to obtain the img directories")
parser.add_argument("-o", "--output-directory", required=True, help="parent directory of the outputs, combine with SUBFOLDERS to obtain the output directories of each set")
parser.add_argument("-sv", "--save-visualization", default="True", help="save prediction result visualization")
parser.add_argument("-ss", "--save-stats", default="True", help="save pred stats")
parser.add_argument("-scs", "--save-combined-stats", default="True", help="save combined pred stats")
parser.add_argument("-sp", "--save-plots", default="True", help="save plots")
parser.add_argument("-ssa", "--save-segment-anything", default="True", help="save segment anything result based on prediction masks")

# TODO: if not save pred, then read from csv


args = parser.parse_args()

# TODO: allow individual options for each set 
args.save_visualization = True if args.save_visualization == "True"  else False
args.save_stats = True if args.save_stats == "True" else False
args.save_combined_stats = True if args.save_combined_stats == "True" else False
args.save_plots = True if args.save_plots == "True" else False
args.save_segment_anything = True if args.save_segment_anything == "True" else False

print("settings: ", args.save_visualization, args.save_stats, args.save_combined_stats, args.save_plots, args.save_segment_anything)

# Load class names
CLASS_NAMES = ['BG']
SUBFOLDERS = [ "詹老師jpg", "世曦", "Google", "詹老師tif"]
# image names of the intersections, no file extension
# IMG_NAMES = ["0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012","0013","0014","0015"]
IMG_NAMES = ["0001","0002","0003","0004","0005"]

# this is for situations where the image name does not match the key in annot.json, e.g. the image is called 0001result.png, and you want key to be 0001 
# a regex to extract the necessary info, and insert it in order (TODO: allow diff order) into the format string. If regex string is empty then no formatting.
ANNOT_FORMATS = [(r"", "{}"), (r"", "{}"), (r"", "{}"),(r"", "{}"), (r"", "{}")]
# (r"(.*)\.", "{}.jpg"), 


IMGS_PARENT_DIR = os.path.normpath(args.images_directory)
OUTPUT_DIR = os.path.normpath(args.output_directory)
# PRED_RESULT_FILE = os.path.normpath(args.output_directory_file)


with open(os.path.normpath(args.labels), 'r') as f:
    CLASS_NAMES.extend(f.read().strip().split('\n'))

print("Label names: ", CLASS_NAMES)

NUM_OF_USER_CLASS_TYPES = len(CLASS_NAMES) - 1

class CustomConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

    DETECTION_MIN_CONFIDENCE = 0.3

    IMAGE_RESIZE_MODE = "none"

def format_name_to_annot_key(input, format_tuple):
    regex_str, format_str = format_tuple

    if regex_str == "":
        return input
    
    matches = re.findall(regex_str, input)

    if len(matches) == 0:
        return input
    
    return format_str.format(*matches)

def load_image_data(img_path, annot_path, format_tuple):
    
    annotation = {}
    with open(annot_path) as f: 
        annotation = json.load(f)

    img_name = os.path.basename(os.path.normpath(img_path))
    key = format_name_to_annot_key(img_name, format_tuple)
    cur_annot = annotation[key]

    # each dict ("x" : ..., "y" : ...) represent a polygon, "all points x" is a list of x coordinates, 
    polygons = [{"x" : item["shape_attributes"]["all_points_x"], "y" : item["shape_attributes"]["all_points_y"], "label" : item["region_attributes"]["label"]} for item in cur_annot["regions"].values()]
    img = skimage.io.imread(img_path)
    # how many rows
    height = img.shape[0]
    width = img.shape[1]

    masks = np.zeros([height, width, len(polygons)], dtype='bool')

    class_ids = []
    for i in range(len(polygons)):
        rr, cc = skimage.draw.polygon(polygons[i]["y"] ,polygons[i]["x"])

        masks[rr,cc, i] = 1

        # masks[rr,cc, i] = 1
        class_ids.append(CLASS_NAMES.index(polygons[i]["label"]))
    b_boxes = mrcnn.utils.extract_bboxes(masks)

    # return: 
    #   masks: masks in the shape [height, width, instances] 
    #   class_ids: a 1D array of class IDs of the instance masks.\
    #   b_boxes: bounding boxes of the masks
    return masks, np.asarray(class_ids, dtype='int32'), b_boxes


config = CustomConfig()
# Load model
model = mrcnn.model.MaskRCNN(mode="inference", model_dir=str(), config=config)
print("\n\nweight path: ",os.path.normpath(args.weight), "\n\n")
model.load_weights(filepath=os.path.normpath(args.weight), by_name=True)
ANNOT_PARENT_PATH = os.path.normpath(args.annot)




fields = ["Img", "GT", "Pred", "Score", "IfValid"]


def scale_combine_pred(imgs_dir, config, annot_path, vis_output_dir):
    """
    Return:
        result: [num of images, num_of_gt_pred_pairs, ("Img", "GT", "Pred", "Score", "IfValid")], rows of gt pred pair.
        img_idx_order: [num of images], records the id of the current image, so as to know which intersection this image contains
        num_of_gts: [num of images, num of class types], output the number of gt of the class type of that image
    """
    result = []
    img_idx_order = []
    total_gt_match = np.array([]).astype(np.int32)
    total_pred_match = np.array([]).astype(np.int32)
    total_thresholds = np.array([])
    total_pred_class_ids = np.array([]).astype(np.int32)
    total_gt_class_ids = np.array([]).astype(np.int32)
    current_subfolder = os.path.basename(os.path.normpath(imgs_dir))
    num_of_gts = []

    for image_name in os.listdir(imgs_dir):
        image_path = os.path.join(imgs_dir, image_name)
        
        if not os.path.isfile(image_path):
            continue

        # removes file extension
        # base_name_wo_file_extension = '.'.join(image_name.split('.')[:-1])
        base_name_wo_file_extension = image_name.split('.')[0]
        # store the image index
        
        # if read file not in the specified list of img names, then not what we are looking for
        try:
            img_idx_order.append(IMG_NAMES.index(base_name_wo_file_extension))
        except:
            continue

        print("Image: ", image_name)
        image =  cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        if(image is None):
            print("filetype not compatible")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # predition and combination of masks
        pred_result = model.detect_multi_scale_and_combine_windowed(image, start_scale = 0.35, end_scale = 1.2, scale_step = 0.05, overlap_threshold=0.35)


        if (args.save_visualization):
            visualize.save_instances_visualization_windowed(image=image, 
                            save_path=os.path.join(vis_output_dir,"vis_"+ base_name_wo_file_extension + ".png"),
                            boxes=pred_result['rois'], 
                            windowed_masks = pred_result['masks'], 
                            class_ids=pred_result['class_ids'], 
                            class_names=CLASS_NAMES, 
                            scores=pred_result['scores'])   

        # if need gt info
        if args.save_stats or args.save_plots:
            gt_masks, gt_class_ids, gt_bboxes = load_image_data(image_path, annot_path, ANNOT_FORMATS[SUBFOLDERS.index(current_subfolder)])
            total_gt_class_ids = np.concatenate([total_gt_class_ids, gt_class_ids])
            total_pred_class_ids = np.concatenate([total_pred_class_ids, pred_result['class_ids']])

            temp_num_of_gts = [0 for _ in range(NUM_OF_USER_CLASS_TYPES)]
            for gt_class_id in gt_class_ids:
                assert gt_class_id > 0
                temp_num_of_gts[gt_class_id - 1] += 1

            num_of_gts.append(temp_num_of_gts)


        if (args.save_stats): 
            # log gt prediction pairs 
            temp_result = utils.compute_gt_pred_pairs(gt_masks = gt_masks,
                                    gt_boxes = gt_bboxes,
                                    gt_class_ids = gt_class_ids,
                                    pred_boxes = pred_result['rois'], 
                                    pred_masks = pred_result['masks'], 
                                    pred_class_ids = pred_result['class_ids'], 
                                    pred_scores = pred_result['scores'],
                                    class_names = CLASS_NAMES,
                                    iou_threshold = 0.3, 
                                    ifWindowed=True)
            
            temp_result = np.array(temp_result)

            image_name_column = np.full((temp_result.shape[0], 1), image_name)
            temp_result = np.concatenate((image_name_column, temp_result), axis=1)
            
            result.append(temp_result)
        

        if args.save_plots:
            precision_recall_dir = os.path.join(vis_output_dir, "plots", "Precision_Recall")
            precision_recall_vs_thresholds_dir = os.path.join(vis_output_dir, "plots", "Precision_Recall_vs_Thresholds")
            gt_match, pred_match, _ , thresholds = utils.compute_matches(gt_bboxes, gt_class_ids, gt_masks,
                                                                pred_result['rois'], pred_result['class_ids'], pred_result['scores'], pred_result['masks'], ifWindowed=True)
            total_gt_match = np.concatenate([total_gt_match, gt_match])
            total_pred_match = np.concatenate([total_pred_match, pred_match])
            total_thresholds = np.concatenate([total_thresholds, thresholds])

            AP, precisions, recalls, _ , thresholds = utils.compute_ap(gt_bboxes, gt_class_ids, gt_masks,
                                          pred_result['rois'], pred_result['class_ids'], pred_result['scores'], pred_result['masks'], ifWindowed=True)
            
            visualize.save_precision_recall_plot(AP, precisions, recalls,
                                                save_path = os.path.join(precision_recall_dir, "precision_recall_"+ base_name_wo_file_extension + ".png"))
            
            visualize.save_precision_recall_vs_thresholds(AP, precisions, recalls, thresholds,
                                                             save_path = os.path.join(precision_recall_vs_thresholds_dir, "precision_recall_vs_thresh_"+ base_name_wo_file_extension + ".png"))

    
    del pred_result

    # save combined plots
    if args.save_plots:
        def comute_ap_with_matches(sorted_pred_matches, sorted_gt_matches) : 
            # plot output dir

            # Compute precision and recall at each prediction box step
            precisions = np.cumsum(sorted_pred_matches > -1) / (np.arange(len(sorted_pred_matches)) + 1)
            recalls = np.cumsum(sorted_pred_matches > -1).astype(np.float32) / len(sorted_gt_matches)

            # Pad with start and end values to simplify the math
            # to compute area under the curve??????????
            precisions = np.concatenate([[0], precisions, [0]])
            recalls = np.concatenate([[0], recalls, [1]])

            # Ensure precision values decrease but don't increase. This way, the
            # precision value at each recall threshold is the maximum it can be
            # for all following recall thresholds, as specified by the VOC paper.
            # since you if precision value increase as recall increases, then there is no real reason why you would choose a lower precision rate with a lower recall rate
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = np.maximum(precisions[i], precisions[i + 1])

            # Compute mean AP over recall range
            # compute area under the curve, the max area is 1, where precision is 1 for all recall rates
            indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
            mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                        precisions[indices])
            
            recalls = recalls[1:-1]
            precisions = precisions[1:-1]

            return mAP, precisions, recalls

        plots_ouput_dir = os.path.join(vis_output_dir, "plots")
        
        # sort by scores 
        sorted_thresholds_idx = np.argsort(total_thresholds)[::-1]
        temp_thresholds = total_thresholds[sorted_thresholds_idx]
        temp_pred_matches = total_pred_match[sorted_thresholds_idx]

        AP, precisions, recalls = comute_ap_with_matches(temp_pred_matches, total_gt_match)

        visualize.save_precision_recall_plot(AP, precisions, recalls,
                                            save_path = os.path.join(plots_ouput_dir, "combined_precision_recall.png"))
        
        visualize.save_precision_recall_vs_thresholds(AP, precisions, recalls, temp_thresholds,
                                                            save_path = os.path.join(plots_ouput_dir, "combined_precision_recall_vs_thresh.png"))
        

        # sort by classes then scores
        # [num of classes, num of indices that corre to this class], indices of predictions that belongs to this class

        pred_class_indices = []
        for i, class_name in enumerate(CLASS_NAMES):
            pred_class_indices.append([])
            for j in range(len(total_pred_class_ids)):
                if CLASS_NAMES[total_pred_class_ids[j]] == class_name:
                    pred_class_indices[i].append(j)

        gt_class_indices = []
        for i, class_name in enumerate(CLASS_NAMES):
            gt_class_indices.append([])
            for j in range(len(total_gt_class_ids)):
                if CLASS_NAMES[total_gt_class_ids[j]] == class_name:
                    gt_class_indices[i].append(j)

        # make a combined graph
        # AP, precisions, recalls become a array of APs, and so on and so forth
        AP = []
        precisions = []
        recalls = []
        thresholds = []

        for i in range(len(CLASS_NAMES)):
            temp_thresholds = total_thresholds[pred_class_indices[i]]
            temp_pred_matches = total_pred_match[pred_class_indices[i]]
            temp_gt_matches = total_gt_match[gt_class_indices[i]]

            # sort by score, descending, does not need to sort gt_matches, as its length is all we need
            sorted_thresholds_idx = np.argsort(temp_thresholds)[::-1]
            temp_thresholds = temp_thresholds[sorted_thresholds_idx]
            temp_pred_matches = temp_pred_matches[sorted_thresholds_idx]

            temp_AP, temp_precisions, temp_recalls = comute_ap_with_matches(temp_pred_matches, temp_gt_matches)
            
            thresholds.append(temp_thresholds)
            AP.append(temp_AP)
            precisions.append(temp_precisions)
            recalls.append(temp_recalls)
        
        # each row shall contain 5 plots
        ncols =  5
        quotient, remainder = divmod(len(CLASS_NAMES), ncols)
        nrows = quotient + 1

        vs_thresholds_fig, vs_thresholds_axes = plt.subplots(nrows, ncols,figsize = (18, 14), constrained_layout = True)
        precision_recall_fig, precision_recall_axes = plt.subplots(nrows, ncols, figsize = (18, 14),  constrained_layout = True)

        # vs_thresholds_fig.tight_layout()
        # precision_recall_fig.tight_layout()

        for row in range(nrows):
            for col in range(ncols):
                # if current number of drawn subplots exceeds len(CLASS_NAMES), break
                cur_class_idx = row * ncols + col
                if(cur_class_idx + 1 > len(CLASS_NAMES)):
                    break
                
                vs_thresholds_axes[row][col].set_title(CLASS_NAMES[cur_class_idx])
                vs_thresholds_axes[row][col].set_xlim(0, 1.1)
                vs_thresholds_axes[row][col].set_ylim(0, 1.1)
                vs_thresholds_axes[row][col].set_xlabel("Thresholds")
                vs_thresholds_axes[row][col].set_ylabel("Rates")
                vs_thresholds_axes[row][col].plot(thresholds[cur_class_idx], precisions[cur_class_idx], label="Precision")
                vs_thresholds_axes[row][col].plot(thresholds[cur_class_idx], recalls[cur_class_idx], label="Recall")
                vs_thresholds_axes[row][col].legend()

                precision_recall_axes[row][col].set_title(CLASS_NAMES[cur_class_idx])
                precision_recall_axes[row][col].set_xlim(0, 1.1)
                precision_recall_axes[row][col].set_ylim(0, 1.1)
                precision_recall_axes[row][col].set_xlabel("Recall")
                precision_recall_axes[row][col].set_ylabel("Precision")
                precision_recall_axes[row][col].plot( recalls[cur_class_idx], thresholds[cur_class_idx])
        
        vs_thresholds_fig.subplots_adjust(wspace=0.5,hspace = 0.5)
        precision_recall_fig.subplots_adjust(wspace=0.5,hspace = 0.5)

        vs_thresholds_fig.savefig(os.path.join(plots_ouput_dir, "vs_thresh_by_class.png"))
        precision_recall_fig.savefig(os.path.join(plots_ouput_dir, "precision_recall_by_class.png"))

    return result, img_idx_order, num_of_gts

# fields : ["Img", "GT", "Pred", "Score", "IfValid"]
#  A predicted instance that correctly matches a ground truth instance / num of predicted instances
def precision_rate(pred_result):
    """
    pred_result: [num of images, ("Img", "GT", "Pred", "Score", "IfValid")]

    returns: 
        img_rate: [num of images, num of classes], [i][j] precision rate of the jth class in the ith image 
        type_rate: [num of classes], [i] precision rate of the ith class, combining the precision rates of the same class across all images 
        total_rate: total precision rate, agregating all prediction of every class across all images
        num_of_each_pred: [num of images, num of classes], [i][j] number of predictions of jth class in the ith image
    
    """
    num_of_each_pred = np.zeros((len(pred_result), NUM_OF_USER_CLASS_TYPES))
    num_of_each_correct_pred = np.zeros((len(pred_result),NUM_OF_USER_CLASS_TYPES))

    img_rate = np.zeros((len(pred_result), NUM_OF_USER_CLASS_TYPES))
    img_idx = 0

    for result in pred_result:
        for row in result:
            if row[2] != "None":
                num_of_each_pred[img_idx][CLASS_NAMES.index(row[2]) - 1] += 1

                # if gt matches pred
                if row[1] == row[2]:
                    num_of_each_correct_pred[img_idx][CLASS_NAMES.index(row[2]) - 1] += 1

        img_idx += 1


    img_rate = num_of_each_correct_pred / num_of_each_pred
    type_rate = np.sum(num_of_each_correct_pred, axis=0) / np.sum(num_of_each_pred, axis=0)
    # [:-1] TEMP SOL
    total_rate = np.sum(num_of_each_correct_pred[:-1]) / np.sum(num_of_each_pred[:-1])

    return img_rate, type_rate, total_rate, num_of_each_pred

# num of ground truth instance that has a correct prediction / num of gt instances
def recall_rate(pred_result, num_of_gts):
    # num_of_each_gt = np.zeros((len(pred_result), NUM_OF_USER_CLASS_TYPES))
    num_of_gts = np.array(num_of_gts)
    num_of_each_correct_pred = np.zeros((len(pred_result), NUM_OF_USER_CLASS_TYPES))
    img_rate = np.zeros((len(pred_result), NUM_OF_USER_CLASS_TYPES))
    
    img_idx = 0
    for result in pred_result:
        for row in result:
            # # if gt
            # if row[1] != "None":
            #     num_of_each_gt[img_idx][CLASS_NAMES.index(row[1]) - 1] += 1

            #     # if gt matches pred
            #     if row[1] == row[2]:
            #         num_of_each_correct_pred[img_idx][CLASS_NAMES.index(row[1]) - 1] += 1

            # if gt matches pred
            if row[1] == row[2]:
                num_of_each_correct_pred[img_idx][CLASS_NAMES.index(row[1]) - 1] += 1

        img_idx += 1



    img_rate = num_of_each_correct_pred / num_of_gts
    type_rate = np.sum(num_of_each_correct_pred, axis=0) / np.sum(num_of_gts, axis=0)
    # [:-1] TEMP SOL
    total_rate = np.sum(num_of_each_correct_pred[:-1]) / np.sum(num_of_gts[:-1])

    return img_rate, type_rate, total_rate, num_of_gts

# def one_minus_precision_rate(pred_result):
        

#     img_rate, type_rate, total_rate, num_of_each_pred= precision_rate(pred_result)
    
#     img_rate = 1 - img_rate
    
#     type_rate = 1 - type_rate

#     total_rate = 1 - total_rate
    
#     # print("output: ",output)
#     return img_rate, type_rate, total_rate, num_of_each_pred

# def one_minus_recall_rate(pred_result):

#     img_rate, type_rate, total_rate, num_of_each_pred = recall_rate(pred_result)
    
#     img_rate = 1 - img_rate
    
#     type_rate = 1 - type_rate

#     total_rate = 1 - total_rate
    
#     # print("output: ",output)
#     return img_rate, type_rate, total_rate, num_of_each_pred

def print_metric_of_set(type_rate ,total_rate, num_of_samples, heading, file=sys.stdout, ifRecall = True):
    # if not recall then precision
    count_name = "GTCount" if ifRecall else "PredCount"

    print(heading, file=file)
    num_of_samples_of_each_class = np.sum(num_of_samples, axis=0)
    print(f"{'綜合機率:':<20} {total_rate:<10.3f}\n", file=file)
    for i in range(0,NUM_OF_USER_CLASS_TYPES - 1):
        true_pos = num_of_samples_of_each_class[i] * type_rate[i]
        if not np.isnan(true_pos):
            true_pos = int(true_pos)
        else:
            true_pos = np.nan
        print( f"{CLASS_NAMES[i + 1]:<20}:{type_rate[i]:<10.3f} || {count_name}:{int(num_of_samples_of_each_class[i]):10d} || TP:{true_pos:10.0f}", file=file)

def print_metric_of_img_type(precision_img_rate_of_each_set , recall_img_rate_of_each_set, precision_num_of_pred_of_each_set, recall_num_of_gt_of_each_set, img_idx_order_of_each_set, file=sys.stdout):
    """
        combines stats of same img from different sets
    """
    
    # [num of images, num of classes]
    combined_precision_img_rate = np.zeros((len(IMG_NAMES), NUM_OF_USER_CLASS_TYPES))
    # [num of images, num of classes]
    combined_recall_img_rate = np.zeros((len(IMG_NAMES), NUM_OF_USER_CLASS_TYPES))

    # [num of images, num of classes]
    combined_precision_num_of_pred = np.zeros((len(IMG_NAMES), NUM_OF_USER_CLASS_TYPES))
    # [num of images, num of classes]
    combined_recall_num_of_gt = np.zeros((len(IMG_NAMES), NUM_OF_USER_CLASS_TYPES))

    for cur_img_idx in range(len(IMG_NAMES)):
        index_of_current_image_of_each_set = []
        for ordering in img_idx_order_of_each_set:
            try:
                index_of_current_image_of_each_set.append(ordering.index(cur_img_idx))
            except ValueError:
                index_of_current_image_of_each_set.append(-1)

        # combine precision rate, recall rate,
        # in range(num of sets)
        for i in range(len(SUBFOLDERS)):
            idx_of_img = index_of_current_image_of_each_set[i]
            # if this image does not exist in this set
            if idx_of_img == -1:
                continue

            combined_precision_img_rate[cur_img_idx] += np.nan_to_num(np.array(precision_img_rate_of_each_set[i][idx_of_img]) * np.array(precision_num_of_pred_of_each_set[i][idx_of_img]))
            combined_recall_img_rate[cur_img_idx] += np.nan_to_num(np.array(recall_img_rate_of_each_set[i][idx_of_img]) * np.array(recall_num_of_gt_of_each_set[i][idx_of_img]))
            combined_precision_num_of_pred[cur_img_idx] += np.nan_to_num(np.array(precision_num_of_pred_of_each_set[i][idx_of_img]))
            combined_recall_num_of_gt[cur_img_idx] += np.nan_to_num(np.array(recall_num_of_gt_of_each_set[i][idx_of_img]))

    
    # [:,:-1] TEMP SOL for removing SpeedLimitMarkings
    total_precision_rate_of_img = np.sum(combined_precision_img_rate[:,:-1], axis=1) / np.sum(combined_precision_num_of_pred[:,:-1], axis=1)
    total_recall_rate_of_img = np.sum(combined_recall_img_rate[:,:-1], axis=1) / np.sum(combined_recall_num_of_gt[:,:-1], axis=1)

    combined_precision_img_rate = combined_precision_img_rate / combined_precision_num_of_pred
    combined_recall_img_rate = combined_recall_img_rate / combined_recall_num_of_gt


    for cur_img_idx in range(len(IMG_NAMES)):
        try:
            index_of_current_image_of_each_set = ordering.index(cur_img_idx)
        except ValueError:
            index_of_current_image_of_each_set = -1
        
        if index_of_current_image_of_each_set == -1:
            continue

        print("Image: ", IMG_NAMES[cur_img_idx],file=file)


        print("(對每個GT正確判斷的機率):",file=file)
        print(f"{'綜合機率:':<20} {total_recall_rate_of_img[cur_img_idx]:<10.3f}\n", file=file)
        for i in range(0, NUM_OF_USER_CLASS_TYPES - 1):
            true_pos = combined_recall_num_of_gt[cur_img_idx][i] * combined_recall_img_rate[cur_img_idx][i]
            if not np.isnan(true_pos):
                true_pos = int(true_pos)
            print( f"{CLASS_NAMES[i + 1]:<20}:{combined_recall_img_rate[cur_img_idx][i]:<10.3f} || GTCount:{int(combined_recall_num_of_gt[cur_img_idx][i]):10d} || TP:{true_pos:10.0f}", file=file)

        print("\n(每個PRED正確判斷的機率):",file=file)
        print(f"{'綜合機率:':<20} {total_precision_rate_of_img[cur_img_idx]:<10.3f}\n", file=file)
        for i in range(0, NUM_OF_USER_CLASS_TYPES - 1):
            true_pos = combined_precision_num_of_pred[cur_img_idx][i] * combined_precision_img_rate[cur_img_idx][i]
            if not np.isnan(true_pos):
                true_pos = int(true_pos)
            print( f"{CLASS_NAMES[i + 1]:<20}:{combined_precision_img_rate[cur_img_idx][i]:<10.3f} || PredCount:{int(combined_precision_num_of_pred[cur_img_idx][i]):10d} || TP:{true_pos:10.0f}", file=file)

        
        print("\n=================================", file=file)


# [num of sets, num of images, num of classes]
recall_img_rate_of_each_set = []

# [num of sets, num of images, num of classes]
precision_img_rate_of_each_set = []

# [num of sets, num of images, num of classes]
precision_num_of_pred_of_each_set = []

# [num of sets, num of images, num of classes]
recall_num_of_gt_of_each_set = []

# [num of sets, num of images]
img_idx_order_of_each_set = []

# prints out stats per set
for i, subfolder_name in enumerate(SUBFOLDERS):
    # dir of each set of images
    images_dir = os.path.normpath(os.path.join(IMGS_PARENT_DIR, subfolder_name))
    # annotation path
    annot_path = os.path.normpath(os.path.join(ANNOT_PARENT_PATH, subfolder_name, "annot.json"))
    # output path
    output_path = os.path.normpath(os.path.join(os.path.normpath(args.output_directory), subfolder_name, "result.txt"))

    vis_output_dir = os.path.normpath(os.path.join(os.path.normpath(args.output_directory), subfolder_name))

    # calculate stats of the current set 
    # result: [num of images, num of gt pred pairs,("Img", "GT", "Pred", "Score", "IfValid")]
    result, img_idx_order, num_of_gts = scale_combine_pred(images_dir, config, annot_path, vis_output_dir)
    img_idx_order_of_each_set.append(img_idx_order)

    if args.save_stats:
        csv_dir = os.path.join(os.path.normpath(args.output_directory), subfolder_name, "csv")
        for i, img_idx in enumerate(img_idx_order):
            result_csv_path = os.path.join(csv_dir, IMG_NAMES[img_idx] + ".csv")
            with open(result_csv_path, "w+") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvwriter.writerows(result[i])

        print('')

        recall_img_rate, recall_type_rate, recall_total_rate, recall_num_of_each_pred = recall_rate(result, num_of_gts)
        recall_img_rate_of_each_set.append(recall_img_rate)
        recall_num_of_gt_of_each_set.append(recall_num_of_each_pred)
        print_metric_of_set(recall_type_rate,recall_total_rate, recall_num_of_each_pred,"\n(對每個GT正確判斷的機率):", ifRecall=True)

        precision_img_rate, precision_type_rate, precision_total_rate, precision_num_of_each_pred = precision_rate(result)
        precision_img_rate_of_each_set.append(precision_img_rate)
        precision_num_of_pred_of_each_set.append(precision_num_of_each_pred)
        print_metric_of_set(precision_type_rate, precision_total_rate, precision_num_of_each_pred,"\n(每個PRED正確判斷的機率):", ifRecall=False)

        # save to output file
        with open(output_path, "w+", encoding='utf-8') as f: 
            print(f"data from images in dir: {images_dir}",file=f)
            print_metric_of_set(recall_type_rate,recall_total_rate,recall_num_of_each_pred,"\n(對每個GT正確判斷的機率):",file=f, ifRecall=True)
            print_metric_of_set(precision_type_rate, precision_total_rate, precision_num_of_each_pred,"\n(每個PRED正確判斷的機率):",file=f, ifRecall=False)

# combine stats per set to obtain stats per image
if args.save_stats and args.save_combined_stats: 
    output_path = os.path.join(os.path.normpath(args.output_directory), "combined_result.txt")
    print("\n每個個別圖資類別的數據\n")
    print_metric_of_img_type(precision_img_rate_of_each_set , recall_img_rate_of_each_set, precision_num_of_pred_of_each_set, recall_num_of_gt_of_each_set, img_idx_order_of_each_set)
    with open(output_path, "w+", encoding='utf-8') as f: 
        print_metric_of_img_type(precision_img_rate_of_each_set , recall_img_rate_of_each_set, precision_num_of_pred_of_each_set, recall_num_of_gt_of_each_set, img_idx_order_of_each_set, file=f)

    