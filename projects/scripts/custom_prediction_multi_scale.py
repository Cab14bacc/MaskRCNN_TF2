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
    IMAGES_PER_GPU = 1


	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
    DETECTION_MIN_CONFIDENCE = 0.7


config = CustomConfig()
# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=str())

# Load the weights into the model.
model.load_weights(filepath=os.path.normpath(args.weight), 
                   by_name=True)


# assumes square mode and scale larger than 1 
def pad_given_scale(image, min_dim, max_dim, scale):
    """
    this scale factor scales each side by that amount, meaning the each side of the original image in the new image will be smaller by a factor of the scale parameter
    e.g. scale = 2 
                 ______
    ori img :   |      |       
                |      |
                |______|

    new img:   ______ ______
              |      |      | 
              | ori  |      |
              |______|______|            
              |      |      | 
              |      |      |
              |______|______|


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
    
    num_of_scaled_image = 15

    scaled_images = []
    for i in range(num_of_scaled_image):
        scaled_images.append(pad_given_scale(image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, 1.0 + i * 0.1))
            
    result = []
    for i in range(num_of_scaled_image):
        r = model.detect([scaled_images[i]], verbose=0)
        result.append(r[0])




    #sorted in order of ('num of class types'), ('num of instances'), ('avg scores')
    avg_scores = []
    num_of_insts = []
    num_of_class_types = []
    for i in range(len(result)):
      avg_scores.append(np.average(result[i]["scores"]))
      num_of_insts.append(len(result[i]["class_ids"]))
      num_of_class_types.append(len(set(result[i]["class_ids"])))
    
    print("number of types of classes", num_of_class_types)
    # sorted_indices = np.argsort(avg_scores)
    # print("sorted indices", sorted_indices)

    dt = np.dtype([('num of class types', int), ('num of instances', int), ('avg scores', int)])
    meta_data_arr = [x for x in zip(num_of_class_types, num_of_insts, avg_scores) if not np.isnan(x[2])]
    meta_data_arr = np.array(meta_data_arr, dtype=dt)
    sorted_indices = np.argsort(meta_data_arr, order=["num of class types", "num of instances", "avg scores"])


    print("Predicted with weights: ", os.path.normpath(args.weight))

    r = result[sorted_indices[-1]]
    for i in range(len(r["rois"])):
        print(CLASS_NAMES[r["class_ids"][i]], r["scores"][i])

    print()
    mrcnn.visualize.display_instances(image=scaled_images[sorted_indices[-1]], 
                                    boxes=r['rois'], 
                                    masks=r['masks'], 
                                    class_ids=r['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                      scores=r['scores'])
    