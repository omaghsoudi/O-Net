import numpy as np
import cv2, os, json
from glob import glob
from copy import deepcopy
from keras.callbacks import BaseLogger
from keras.preprocessing.image import ImageDataGenerator



def get_image_info(obj, Path, Image_folder):
    # This function checks type of images and the number of images
    Image_Path = os.path.join(Path, Image_folder)
    Image_Path = os.path.join(Image_Path, "*"+obj.image_format)

    Images_files = sorted(glob(Image_Path))

    # get one sample of images
    if len(Images_files)>=1:
        Image_sample = cv2.imread(Images_files[0], -1)
    else:
        Image_sample = cv2.imread(Images_files, -1)


    target_size = (Image_sample.shape[0], Image_sample.shape[1])
    num_image = len(Images_files)

    image_dimension = 1
    image_color_mode = "grayscale"
    if len(Image_sample.shape)>2:
        image_dimension = Image_sample.shape[2]
        image_color_mode = "rgb"

    return(target_size, image_dimension, num_image, image_color_mode)



def adjustData(obj, image, mask):
    # image preprocessing steps can be added here
    if len(image.shape)!=4:
        temp = deepcopy(image)
        image = np.zeros(image.shape + (1, ))
        image[..., 0] = temp


    # mask preprocessing steps can be added from here
    # mask is considerd as a grayscale image
    mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]

    # add a dimension for mask classes
    if len(mask.shape) == 3:
        new_mask = np.zeros(mask.shape + (obj.num_class, ))

    # make mask to have as many needed classes
    for i in range(obj.num_class):
        new_mask[mask==i, i] = 1


    return (image, new_mask)



def trainGenerator(obj, Path, Seed=10, Mode="train"):
    if Mode=="train":
        # for training apply augmentation
        image_datagen = ImageDataGenerator(**obj.aug_dict)
    else:
        # No augmentation for validation and testing (if needed use the tarin mode)
        image_datagen = ImageDataGenerator()


    image_generator = image_datagen.flow_from_directory(
        Path,
        classes = [obj.image_folder],
        class_mode = None,
        color_mode = obj.image_color_mode,
        target_size = obj.target_size,
        batch_size = obj.batch_size,
        seed = Seed)


    # for the masks
    mask_datagen = ImageDataGenerator(**obj.aug_dict)
    mask_generator = mask_datagen.flow_from_directory(
        Path,
        classes = [obj.mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = obj.target_size,
        batch_size = obj.batch_size,
        seed = Seed)


    # yelding images and any required preprocessing
    train_generator = zip(image_generator, mask_generator)
    for (image, mask) in train_generator:
        # here you can adjust images (like scaling or any other preprocessing needed)
        (image, mask) = adjustData(obj, image, mask)

        yield (image, mask)



# saving JSON file to keep track of training and validation for each epoch
class Monitor_Training(BaseLogger):
    def __init__(self, jsonPath):
        # store the output path for the JSON serialized file and the starting epoch
        super(Monitor_Training, self).__init__()
        self.jsonPath = jsonPath

        if os.path.exists(self.jsonPath):
            self.History = os.remove(self.jsonPath)


    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.History = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.History = json.loads(open(self.jsonPath).read())


    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        for (Key, Value) in logs.items():
            l = self.History.get(Key, [])
            l.append(Value)
            self.History[Key] = l

        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.History))
            f.close()
