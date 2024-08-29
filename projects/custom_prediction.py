import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
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
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


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


# load weights once, display multiple imgs  
while True:
    addr = input("\n\nenter path to image ('q' to quit): ") 

    if(addr == 'q' or addr == 'Q' ):
        print("quitting............")
        break

    if(not os.path.exists(addr)):
        print("file does not exist")
        continue
    
    image = cv2.imread(os.path.normpath(addr))
    if(image is None):
        print("filetype not compatible")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
        
    print("Predicted with weights: ", os.path.normpath(args.weight))
    for i in range(len(r["rois"])):
        print(CLASS_NAMES[r["class_ids"][i]], r["scores"][i])

    mrcnn.visualize.display_instances(image=image, 
                                    boxes=r['rois'], 
                                    masks=r['masks'], 
                                    class_ids=r['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=r['scores'])    