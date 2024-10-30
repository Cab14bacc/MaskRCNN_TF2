import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import argparse


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
    IMAGES_PER_GPU = 4


	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_RESIZE_MODE = "none"


config = CustomConfig()
# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=str())

# Load the weights into the model.
model.load_weights(filepath=os.path.normpath(args.weight), 
                   by_name=True)



while True:
    addr = input("\n\nenter path to image ('q' to quit): ") 
    addr = addr.strip('"')
    if(addr == 'q' or addr == 'Q' ):
        print("quitting............")
        break

    addr = os.path.normpath(addr)

    if(not os.path.exists(addr)):
        print("file does not exist")
        continue

    image = cv2.imdecode(np.fromfile(addr, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    if(image is None):
        print("filetype not compatible")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # for i in range(num_of_scaled_image):
    #     cv2.imshow(str(i), scaled_images[i])
    #     cv2.waitKey(0)  
    #     cv2.destroyAllWindows() 
    
    # result = []
    # for i in range(num_of_scaled_image):
    #     r = model.detect([scaled_images[i]], verbose=0)
    #     result.append(r[0])

    result = model.detect_multi_scale_and_combine(image, verbose=0)

    
    print("displaying instances")
    mrcnn.visualize.display_instances(image=image, 
                                    boxes=result['rois'], 
                                    masks=result['masks'], 
                                    class_ids=result['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=result['scores'])
    

    