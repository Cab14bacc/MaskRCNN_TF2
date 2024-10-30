import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize as visualize
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from skimage.measure import find_contours


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
parser.add_argument("-l", "--labels", required=True, help="location of the list of labels")
parser.add_argument("-w", "--weight", required=True, help="weight used for prediction")
parser.add_argument("-s", "--source-dir", required=True, help="source dir of images")
parser.add_argument("-o", "--output-dir", required=True, help="output directory")

args = parser.parse_args()

CLASS_NAMES = ['BG']
IMG_DIR = os.path.normpath(args.source_dir)
OUTPUT_DIR = os.path.normpath(args.output_dir)
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


config = CustomConfig()

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=str())

# Load the weights into the model.
model.load_weights(filepath=os.path.normpath(args.weight), 
                   by_name=True)


    
print("Predicting with weights: ", os.path.normpath(args.weight))


def save_instances_visualization(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), save_path=None, ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or visualize.random_colors(len(class_names))

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=8, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Figure saved to {save_path}")



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





# load weights once, display multiple imgs  
for image_name in os.listdir(IMG_DIR):
    image_path = os.path.join(IMG_DIR, image_name)

    if not os.path.isfile(image_path):
        continue
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if(image is None):
        print("filetype not compatible")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    num_of_scaled_image = 20

    scaled_images = []
    for i in range(num_of_scaled_image):
        scaled_images.append(pad_given_scale(image,config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, scale = 1.0 + i * 0.05))

    # for i in range(num_of_scaled_image):
    #     cv2.imshow(str(i), scaled_images[i])
    #     cv2.waitKey(0)  
    #     cv2.destroyAllWindows() 
    
    result = []
    for i in range(num_of_scaled_image):
        r = model.detect([scaled_images[i]], verbose=0)
        result.append(r[0])

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

    # avg_scores = np.nan_to_num(avg_scores, nan= -1)
    meta_data_arr = [x for x in zip(num_of_class_types, num_of_insts, avg_scores) if x[0] != 0 and not np.isnan(x[2])]
    meta_data_arr = np.array(meta_data_arr, dtype=dt)
    sorted_indices = np.argsort(meta_data_arr, order=["num of class types", "num of instances", "avg scores"])


    print("Predicted with weights: ", os.path.normpath(args.weight))

    r = result[sorted_indices[-1]]
    for i in range(len(r["rois"])):
        print(CLASS_NAMES[r["class_ids"][i]], r["scores"][i])

    print()
    save_instances_visualization(image=scaled_images[sorted_indices[-1]], 
                                save_path=os.path.join(OUTPUT_DIR,"fig_"+ image_name.split('.')[0] + ".png"),
                                boxes=r['rois'], 
                                masks=r['masks'], 
                                class_ids=r['class_ids'], 
                                class_names=CLASS_NAMES, 
                                scores=r['scores'])   