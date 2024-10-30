
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
import imgaug



parser = argparse.ArgumentParser(description="custom object detection")
# parser.add_argument("-f", "--file", required=True, help="target image")
# parser.add_argument("-a", "--annot", required=True, help="annotation json")
# parser.add_argument("-l", "--labels", required=True, help="labels.txt")
# parser.add_argument("-w", "--weight", required=True, help="weight used for prediction")

args = parser.parse_args()

CLASS_NAMES = ['BG',
               "CrossWalk",
                "YellowGrid",
                "FArrow",
                "RArrow",
                "LArrow",
                "FRArrow",
                "FLArrow",
                "LRArrow",
                "FLRArrow",
                "Stopline",
                "ScooterWaitArea",
                "ScooterWaitTurnArea",
                "SpeedLimitMarking"]
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
def pad_given_scale(image, min_dim, max_dim, scale):
    """
    this function make sures img after resize is scaled each side with the specify value
    namely maintaining aspect ratio, the image after preprocess will be scaled by a factor of the "scale" argument comparing to if you would not touch the image and send it to preprocess

    image: image np array
    min_dim : minimum dimension during resize
    max_dim: max dim during resize
    scale: the scale you want to apply, has to be > 1, since if we allow scale to be larger than 1, a crop not padding will be needed
    

    padding: (top, bottom, left, right), top and left is always 0
    """
    img_shape = image.shape
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
            short_side_scale = min_dim / short_side
            new_long_side = long_side * short_side_scale
            if new_long_side > max_dim:
                new_long_side = max_dim
                padding = long_side * (scale - 1)
            elif new_long_side <= max_dim:
                padding = (max_dim - new_long_side)/short_side_scale + ((max_dim - new_long_side)/short_side_scale + long_side) * (scale - 1) 
        else: #short_side >= min_dim:
            padding = (max_dim - long_side) + max_dim * (scale - 1)
            
    
    if if_height_longer:
        padding =  (0, int(padding), 0 , 0)
    else:
        padding = (0, 0, 0, int(padding))
    
    return cv2.copyMakeBorder(image, *padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])



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

    

# load the input image, convert it from BGR to RGB channel
image = cv2.imdecode(np.fromfile(os.path.normpath(r"D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資\Google\0001.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load the input image, convert it from BGR to RGB channel
image2 = cv2.imdecode(np.fromfile(os.path.normpath(r"D:\Documents\GitRepos\RoadMaskRCNNAnnotations\測試\測試圖資\世曦\0001.jpg"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


image = np.array(image, dtype = np.uint8)

config =  CustomConfig()
padded_image = pad_given_scale(image,config.IMAGE_MIN_DIM,config.IMAGE_MAX_DIM,  2)

augmentation = iaa.Affine(scale=2)
resized_image = augmentation(image = image)

print("ori img before resize", image.shape)
print("padded img before resize", padded_image.shape)

aug = iaa.Sequential([
    # iaa.OneOf([iaa.Affine(rotate=(0, 60)),
    #             iaa.Affine(rotate=(60,120)),
    #             iaa.Affine(rotate=(120, 180)),
    #             iaa.Affine(rotate=(180, 240)),
    #             iaa.Affine(rotate=(240,300)),
    #             iaa.Affine(rotate=(300,360)),
    #             ]),
    # iaa.Affine(translate_px={"x": (-400, 400), "y": (-400, 400)})
    iaa.Affine(translate_px={"x": 400, "y": 400})
])    
augmented_image, augmented_image_window , _ , _ , _ = utils.resize_image(image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
image, window , _ , _ , _ = utils.resize_image(image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
image2, window2 , _ , _ , _ = utils.resize_image(image2, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
padded_image, padded_image_window , _ , _ , _ = utils.resize_image(padded_image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)
resized_image, resized_image_window , _ , _ , _ = utils.resize_image(resized_image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE)

# if training, augmentation happens after util.resize_image
det = aug.to_deterministic()
augmented_image = det.augment_image(augmented_image)



# window (y1, x1, y2, x2)
print("ori image shape", image.shape)
print("padded image shape", padded_image.shape)
cv2.rectangle(image, (window[1], window[0]), (window[3], window[2]), (0,255,0) , 2)
cv2.rectangle(padded_image, (padded_image_window[1], padded_image_window[0]), (padded_image_window[3], padded_image_window[2]), (0,255,0) , 2)
cv2.rectangle(augmented_image, (augmented_image_window[1], augmented_image_window[0]), (augmented_image_window[3], augmented_image_window[2]), (0,255,0) , 2)
print("image window", window[2] - window[0],window[3] - window[1] )
print("padded image window", padded_image_window[2] - padded_image_window[0], padded_image_window[3] - padded_image_window[1])
cv2.imshow("ori image", image)
cv2.imshow("ori image2", image2)
cv2.imshow("padded image", padded_image)
cv2.imshow("resized image", resized_image)
cv2.imshow("augemented image", augmented_image)

# gt_masks, gt_class_ids, gt_b_boxes = load_image_data(image_path, os.path.normpath(r"C:\Users\Leo\Documents\GitRepos\MaskRCNN_TF2\projects\TESTS\TEST5\annot.json"))






# MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
#                     "Fliplr", "Flipud", "CropAndPad",
#                     "Affine", "PiecewiseAffine"]

# def hook(images, augmenter, parents, default):
#     """Determines which augmenters to apply to masks."""
#     return augmenter.__class__.__name__ in MASK_AUGMENTERS

# gt_masks = np.transpose(gt_masks, (2, 0, 1))
# # Create an empty list to store augmented masks

# # Augment each mask individually
# for i in range(gt_masks.shape[0]):
#     gt_masks[i,:,:] = det.augment_image(gt_masks[i,:,:], hooks=imgaug.HooksImages(activator=hook))

# # Stack augmented masks back together
# gt_masks = np.transpose(gt_masks, (1, 2, 0))


# gt_b_boxes = utils.extract_bboxes(gt_masks)


# # print(gt_masks)
# # Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image, 
#                                   boxes=gt_b_boxes, 
#                                   masks=gt_masks, 
#                                   class_ids=gt_class_ids, 
#                                   class_names=CLASS_NAMES, 
#                                   scores=np.zeros(len(gt_class_ids)))

        

           
cv2.waitKey(0)  
cv2.destroyAllWindows() 


    


