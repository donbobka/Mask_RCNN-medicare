"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import skimage.io
import skimage.transform
from scipy.spatial import distance as dist
import imgaug as ia
from imgaug import augmenters as iaa
import glob
import json

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR='./'

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class MedicareConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "medicare"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    
    IMAGES_PER_GPU = 2
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    def augumentations(self):
        return iaa.SomeOf((0, 5), [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(shear=(-16, 16)),
                iaa.Multiply((0.8, 1.5)),
                iaa.GaussianBlur(sigma=(0.0, 5.0)),
                iaa.Crop(percent=(0, 0.25))
            ])

class MedicareInferenceConfig(MedicareConfig):
    IMAGES_PER_GPU = 1

class MedicareDataset(utils.Dataset):
    def load_medicare(self, dataset_dir, subset):
        """Load a subset of the medicare dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("medicare", 1, "medicare")

        dataset_dir = os.path.join(dataset_dir, subset, 'images')
        image_ids = next(os.walk(dataset_dir))[2]

        # Add images
        for image_filename in image_ids:
            image_id = os.path.splitext(image_filename)[0]
            self.add_image(
                "medicare",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_filename))

    def load_labelme_medicare(self, dataset_dir, subset):
        """Load a subset of the medicare dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("medicare", 1, "medicare")

        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Add images
        for annonation_filename in glob.iglob(os.path.join(dataset_dir, '*.json')):
            json_file = open(annonation_filename, 'r')
            annotation = json.load(json_file)
            json_file.close

            image_id = os.path.splitext(os.path.basename(annonation_filename))[0]
            self.add_image(
                "medicare",
                image_id=image_id,
                path=os.path.join(dataset_dir, annotation['imagePath']),
                annotation=annotation)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        if 'annotation' in info:
            return self.load_annotation_mask_for_image(image_id)
        return self.load_image_mask_for_image(image_id)
        

    def load_image_mask_for_image(self, image_id):
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        image_mask_path = os.path.join(mask_dir, "{}.png".format(info['id']))
        if not os.path.isfile(image_mask_path):
            return None

        mask = []
        m = skimage.io.imread(image_mask_path).astype(np.bool)
        mask.append(m)
        
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_annotation_mask_for_image(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['annotation']['shapes']
        width = info['annotation']['imageWidth']
        height = info['annotation']['imageHeight']

        # We have only one mask, because we have only medicare class
        mask = np.zeros([height, width, 1], dtype=np.uint8)
        for i, shape in enumerate(shapes):
            if shape['shape_type'] == 'polygon':
                points = np.array([shape['points']], dtype=np.int32)
                mask[:, :, i:i + 1] = cv2.fillPoly(mask[:, :, i:i + 1].copy(), points, (255,255,255))

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "medicare":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MedicareDataset()
    if args.dataset_type == 'default':
        dataset_train.load_medicare(args.dataset, "train")
    elif args.dataset_type == 'labelme':
        dataset_train.load_labelme_medicare(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MedicareDataset()
    if args.dataset_type == 'default':
        dataset_val.load_medicare(args.dataset, "val")
    elif args.dataset_type == 'labelme':
        dataset_val.load_labelme_medicare(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=config.augumentations(),
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=config.augumentations(),
                layers='all')

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def crop_rotated_rect(orig, rotated_rect):
    tl, dim, angle = rotated_rect
    width, height = dim
    
    rect_pts = order_points(cv2.boxPoints(rotated_rect))
    rect_pts= np.roll(rect_pts, 1, axis=0)
    
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect_pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")
    
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warp = cv2.warpPerspective(orig, M, (int(maxWidth), int(maxHeight)))

    if maxWidth < maxHeight:
        warp = skimage.transform.rotate(warp, 90, resize=True)

    return warp

def crop_medicare(model, image):
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    if r["masks"].shape[2] == 0:
        return None
    
    mask = r["masks"][:, :, 0]
    mask_image = np.array(np.uint8(mask * 255))
    contours, hierarchy = cv2.findContours(mask_image, 1, 2)
    rotated_rect = cv2.minAreaRect(contours[0])
    return crop_rotated_rect(image, rotated_rect)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect medicare.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset-type', required=False,
                        default='default',
                        metavar='default|labelme',
                        help='Directory of the Balloon dataset')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MedicareConfig()
    else:
        class InferenceConfig(MedicareConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
