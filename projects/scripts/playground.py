
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import mrcnn.utils as utils
import cv2
import os
import argparse
import skimage
import numpy as np
import json
from imgaug import augmenters as iaa



parser = argparse.ArgumentParser(description="custom object detection")
parser.add_argument("-f", "--file", help="target image")
parser.add_argument("-a", "--annot", help="annotation json")
parser.add_argument("-l", "--labels", help="labels.txt")
parser.add_argument("-w", "--weight", help="weight used for prediction")

args = parser.parse_args()

CLASS_NAMES = ['BG']
MODEL_LOGS_DIR = str()


# with open(os.path.normpath(args.labels), 'r') as f:
#     CLASS_NAMES.extend(f.read().split('\n'))

# print("label names: ", CLASS_NAMES)

class CustomConfig(mrcnn.config.Config):
    NAME = "custom_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = len(CLASS_NAMES)
    
    STEPS_PER_EPOCH = 131

# assumes square mode
# this outputs the padding needed to make the image after resize preprocessing in the prediction / training scale according to the parameter
def cal_padding_given_scale(min_dim, max_dim, img_shape, scale):
    """
    this function make sures img after resize is scaled each side with the specify value
    meaning maintaining aspect ratio, the image after preprocess will be scaled by a factor of the "scale" argument comparing to if you would not touch the image and send it to preprocess

    min_dim : minimum dimension during resize
    max_dim: max dim during resize
    img_shape: the original image shape before resize
    scale: the scale you want to apply
    

    padding: (top, bottom, left, right), top and left is always 0
    """
    if_height_longer = False
    if img_shape[0] > img_shape[1]:
        if_height_longer = True
        long_side = img_shape[0]        
        short_side = img_shape[1]
    else:
        long_side = img_shape[1]
        short_side = img_shape[0]
    
    if long_side > max_dim:
        padding = long_side * (scale - 1)
    else: #long_side <= max_dim:
        if short_side < min_dim:
            temp_scale = min_dim / short_side
            new_long_side = long_side * temp_scale
            if new_long_side > max_dim:
                new_long_side = max_dim
                padding = long_side * (scale - 1)
            elif new_long_side <= max_dim:
                padding = (max_dim - new_long_side)/temp_scale + ((max_dim - new_long_side)/temp_scale + long_side) * (scale - 1) 
        else: #short_side >= min_dim:
            padding = (max_dim - long_side) + max_dim * (scale - 1)
            
    
    if if_height_longer:
        return (0, int(padding), 0 , 0)
    else:
        return (0, 0, 0, int(padding))
    

# load the input image, convert it from BGR to RGB channel
image = cv2.imdecode(np.fromfile(os.path.normpath(args.file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# annot_path = os.path.normpath(args.annot)
image_path= os.path.normpath(args.file)

# image = np.array(image, dtype = np.uint8)

config =  CustomConfig()

padding = cal_padding_given_scale(config.IMAGE_MIN_DIM,config.IMAGE_MAX_DIM, image.shape, 2)
padded_image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])\

augmentation = iaa.Sequential([iaa.HistogramEqualization(), iaa.Affine(rotate=60)])

augmented_image = augmentation(image = image)

print("ori img before resize", image.shape)
print("padded img before resize", padded_image.shape)
print("augmented img before resize", augmented_image.shape)

image, window , _ , _ , _ = utils.resize_image(image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
padded_image, padded_image_window , _ , _ , _ = utils.resize_image(padded_image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
# window (y1, x1, y2, x2)
print("ori image shape", image.shape)
print("padded image shape", padded_image.shape)
cv2.rectangle(image, (window[1], window[0]), (window[3], window[2]), (0,255,0) , 2)
cv2.rectangle(padded_image, (padded_image_window[1], padded_image_window[0]), (padded_image_window[3], padded_image_window[2]), (0,255,0) , 2)
print("image window", window[2] - window[0],window[3] - window[1] )
print("padded image window", padded_image_window[2] - padded_image_window[0], padded_image_window[3] - padded_image_window[1])
cv2.imshow("ori image", image)
cv2.imshow("padded image", padded_image)
cv2.imshow("augmented image",augmented_image)

cv2.waitKey(0)  
cv2.destroyAllWindows() 



        

           
    


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


# gt_masks, gt_class_ids, gt_b_boxes = load_image_data(image_path, annot_path)


# # Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image, 
#                                   boxes=gt_b_boxes, 
#                                   masks=gt_masks, 
#                                   class_ids=gt_class_ids, 
#                                   class_names=CLASS_NAMES, 
#                                   scores=np.zeros(len(gt_class_ids)))

