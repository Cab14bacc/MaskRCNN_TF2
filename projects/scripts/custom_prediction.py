import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.utils
import mrcnn.visualize
import cv2
import os
import numpy as np
import argparse
from imgaug import augmenters as iaa


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
# parser.add_argument("-f", "--file", required=True, help="target image")
# parser.add_argument("-s", "--source", required=True, help="project folder")
parser.add_argument("-l", "--labels", required=True, help="location of the list of labels")
parser.add_argument("-w", "--weight", required=True, help="weight used for prediction")

args = parser.parse_args()

CLASS_NAMES = ['BG']
# DATASET_DIR = args.source
# MODEL_LOGS_DIR = os.path.join(DATASET_DIR, "logs")
# print("project folder: ", DATASET_DIR)


with open(os.path.normpath(args.labels), 'r') as f:
    CLASS_NAMES.extend(f.read().strip().split('\n'))

print("label names: ", CLASS_NAMES)

class CustomConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
    DETECTION_MIN_CONFIDENCE = 0.7


# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=CustomConfig(),
                             model_dir=str())

# Load the weights into the model.
model.load_weights(filepath=os.path.normpath(args.weight), 
                   by_name=True)




# display a single image
# # load the input image, convert it from BGR to RGB channel
# image = cv2.imread(os.path.normpath(args.file))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Perform a forward pass of the network to obtain the results
# r = model.detect([image], verbose=0)

# # Get the results for the first image.
# r = r[0]


# # Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image, 
#                                   boxes=r['rois'], 
#                                   masks=r['masks'], 
#                                   class_ids=r['class_ids'], 
#                                   class_names=CLASS_NAMES, 
#                                   scores=r['scores'])
def scale_img(image, scale):
    h,w = image.shape[:2]
    new_h = int(((scale * h) // 64) * 64)
    new_w = int(((scale * w) // 64) * 64)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, (new_h, new_w)

def unscale_masks(masks, ori_img_shape):
    output_mask = np.zeros((*ori_img_shape, masks.shape[-1])).astype(np.uint8)
    masks = masks.astype(np.uint8)
    height, width = ori_img_shape
    for i in range(masks.shape[-1]):

        output_mask[:, :, i] = cv2.resize(masks[:,:,i], (width, height), interpolation=cv2.INTER_AREA)

    return output_mask.astype(np.bool_)

def unscale_rois(rois, ori_shape, new_shape):
    inverse_scale = (ori_shape[0]/new_shape[0], ori_shape[1]/new_shape[1])
    for i in range(rois.shape[0]):
        rois[i] = rois[i] * np.array([*inverse_scale, *inverse_scale])

    return rois
                
                


ifNone = True
ifAug = True

# load weights once, display multiple imgs  
while True:
    addr = input("\n\nenter path to image ('q' to quit): ") 
    addr = addr.strip('"')

    if(addr == 'q' or addr == 'Q' ):
        print("quitting............")
        break
    
    if(addr == "load"):
        addr = input("weight path:")
        model.load_weights(filepath=os.path.normpath(addr), 
                   by_name=True)
        continue
    
    if(addr == "mode"):
        ifNone = not ifNone
        print("None Mode" if ifNone else "Square Mode")
        continue
    
    
    if(addr == "aug"):
        ifAug = not ifAug
        print("aug activated" if ifAug else "no aug")
        continue

    addr = os.path.normpath(addr)

    if(not os.path.exists(addr)):
        print("file does not exist")
        continue

    image = cv2.imdecode(np.fromfile(addr, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    if(image is None):
        print("filetype not compatible")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    start_scale = float(input("start scale: "))
    end_scale = float(input("end scale: "))
    scale_step = float(input("scale step: "))
    scaled_img, new_shape = scale_img(image, scale=end_scale)
    aug = iaa.Sequential([
        iaa.Affine(scale = end_scale)
    ])    
    det = aug.to_deterministic()
    augmented_img = det.augment_image(image)


    try:
        # r = model.detect_multi_scale(image, start_scale = scale, end_scale = scale, scale_step = 0, verbose=0)
        # r = model.detect([scaled_image], verbose=0)
        if ifNone:
            if ifAug:
                r = model.detect_multi_scale_and_combine_windowed(augmented_img, start_scale, end_scale, scale_step, 0.9, verbose=0)
            else:
                r = model.detect_multi_scale_and_combine_windowed(image, start_scale, end_scale, scale_step, 0.9, verbose=0)
        else:
            if ifAug:
                r = model.detect([augmented_img], verbose=0)
                r = r[0]
                r["masks"] = mrcnn.utils.minimize_masks_windowed(r["rois"], r["masks"])
            else:
                r = model.detect([scaled_img], verbose=0)
                r = r[0]
                r['rois'] = unscale_rois(r['rois'], image.shape[:2], new_shape)
                r["masks"] = mrcnn.utils.minimize_masks_windowed(r["rois"], r["masks"])
                r["masks"] = unscale_masks(r["masks"], ori_img_shape=image.shape[:2])
    except:
        print("failed")
        continue
    # r = r[0]
        
    print("Predicted with weights: ", os.path.normpath(args.weight))
    for i in range(len(r["rois"])):
        print(CLASS_NAMES[r["class_ids"][i]], r["scores"][i])


    
    # r["masks"] = mrcnn.utils.minimize_masks_windowed(r["rois"], r["masks"])

    # r["masks"] = mrcnn.utils.expand_masks_windowed(r["rois"], r["masks"], image_shape = image.shape[:2])  
    if ifAug:
        mrcnn.visualize.display_instances_windowed(image=augmented_img, 
                                        boxes=r['rois'], 
                                        windowed_masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])
    else: 
        mrcnn.visualize.display_instances_windowed(image=image, 
                                        boxes=r['rois'], 
                                        windowed_masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])