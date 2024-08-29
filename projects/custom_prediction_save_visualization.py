import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize as visual
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


# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=CustomConfig(),
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
        auto_show = True

    # Generate random colors
    colors = colors or visual.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

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
            masked_image = visual.apply_mask(masked_image, mask, color)

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
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")


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

    r = model.detect([image], verbose=0)
    r = r[0]

    print("current image:", image_path)
    save_instances_visualization(image=image, 
                                 save_path=os.path.join(OUTPUT_DIR,"fig_"+ image_name.split('.')[0] + ".png"),
                                    boxes=r['rois'], 
                                    masks=r['masks'], 
                                    class_ids=r['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=r['scores'])   
