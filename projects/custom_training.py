import os
import argparse
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import json
import skimage
import skimage.draw
import skimage.io

from mrcnn.visualize import display_instances
import mrcnn.utils
import mrcnn.config
import mrcnn.model


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





parser = argparse.ArgumentParser(description="custom transfer training for object detection")
parser.add_argument("-s", "--source", required=True, help="source of the project folder")
parser.add_argument("-w", "--pretrained-weight", required=True, help="pretrained weight")


args = parser.parse_args()

CLASS_NAMES = ['BG']
DATASET_DIR = os.path.normpath(args.source)
MODEL_LOGS_DIR = os.path.join(DATASET_DIR, "logs")

with open(os.path.join(DATASET_DIR, "labels.txt"), 'r') as f:
    CLASS_NAMES.extend(f.read().strip().split('\n'))

class CustomDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        for i in range(len(CLASS_NAMES) - 1):
            self.add_class("dataset", i + 1, CLASS_NAMES[i + 1])
        

        img_dir = ""
        annot_dir = os.path.join(dataset_dir, "annot.json")

        if is_train:
            img_dir = os.path.join(dataset_dir, "train_images")
        elif not is_train:
            img_dir = os.path.join(dataset_dir, "val_images")
            
        annotation = {}
        with open(annot_dir) as f: 
            annotation = json.load(f)

        for filename in os.listdir(img_dir):
            # take file name as image id
            image_id = filename.split(".")[0]
            img_path = os.path.join(img_dir, filename)
        
            img = skimage.io.imread(img_path)
            # how many rows
            height = img.shape[0]
            width = img.shape[1]

            cur_annot = annotation[filename]
            # each dict ("x" : ..., "y" : ...) represent a polygon, "all points x" is a list of x coordinates, 
            polygons = [{"x" : item["shape_attributes"]["all_points_x"], "y" : item["shape_attributes"]["all_points_y"], "label" : item["region_attributes"]["label"]} for item in cur_annot["regions"].values()]
            self.add_image('dataset', image_id=image_id, path=img_path, width=width, height=height, polygons=polygons)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        # this image_id is internal
        info = self.image_info[image_id]
        polygons = info["polygons"]
        width = info["width"]
        height = info["height"]

        masks = np.zeros([height, width, len(polygons)], dtype='uint8')

        class_ids = []
        for i in range(len(polygons)):
            rr, cc = skimage.draw.polygon(polygons[i]["y"] ,polygons[i]["x"])
            masks[rr,cc, i] = 1
            class_ids.append(self.class_names.index(polygons[i]["label"]))

        # return: 
        #   masks: masks in the shape [height, width, instances] 
        #   class_ids: a 1D array of class IDs of the instance masks.
        return masks, np.asarray(class_ids, dtype='int32')

class CustomConfig(mrcnn.config.Config):
    NAME = "custom_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = len(CLASS_NAMES)

    STEPS_PER_EPOCH = 200


# Train
train_dataset = CustomDataset()
train_dataset.load_dataset(dataset_dir=DATASET_DIR, is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = CustomDataset()
validation_dataset.load_dataset(dataset_dir=DATASET_DIR, is_train=False)
validation_dataset.prepare()

# Model Configuration
custom_config = CustomConfig()
custom_config.display()


#Display Annotations

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
    
    


# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                            model_dir=MODEL_LOGS_DIR, 
                            config=custom_config)

# load pretrained weights
model.load_weights(filepath=args.pretrained_weight, 
                by_name=True, 
                exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# finetune all layers 
model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=custom_config.LEARNING_RATE, 
            epochs=100, 
            layers='all')

output_weights_path = os.path.join(DATASET_DIR, "output_weights")
output_weight_name = os.path.basename(os.path.normpath(DATASET_DIR)) + "_" +  datetime.now().strftime("%Y-%m-%d-%H-%M") + ".h5"
model_path = os.path.normpath(os.path.join(output_weights_path, output_weight_name))
model.keras_model.save_weights(model_path)