# argparse is seprated to avoid extra loading if help needed
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-nc", "--num_classes", type=int, default=3,
                help="Number of classes for segmentation.")

ap.add_argument("-i", "--dataset",
                default="/home/ohm/Desktop/data/train",
                help="Path to input training dataset; "+
                "it should have image and label folders.")

ap.add_argument("-v", "--validation_path",
                default="0",
                help="Path to validation images. If it this is provided, "
                "then, it will be used. \ "
                "0 means no validation and just train network.")

ap.add_argument("-o", "--output",
                default="/home/ohm/Desktop/output",
                help="Path to output to save the results.")

ap.add_argument("-bs", "--batch_size", type=int, default=3,
                help="The size of training batches.")

ap.add_argument("-ne", "--num_epoch", type=int, default=50,
                help="Number of epochs for training.")

ap.add_argument("-act", "--Activation", default="softmax",
                help="It can be softmax or sigmoid based on training mode.")

ap.add_argument("-ss", "--steps_saved", type=int, default=2,
                help="The number of steps that model was saved.")

ap.add_argument("-mt", "--model_type", default="ONet",
                help="Which model to be used, it can be: "+
                "ONet, PSPNet, UNet.")

ap.add_argument("-bb", "--backbone", default="resnet_encoder",
                help="Which backbone will be used. it can be: "+
                "resnet_encoder, vgg_encoder, simple_encoder.")

ap.add_argument("-if", "--image_format", default=".png",
                help="What image format to be used for saving or loading images?")

ap.add_argument("-ifs", "--image_final_size", type=int, default=-1,
                help="Final image size of network, anything above -1 will be set.")


args = vars(ap.parse_args())


# global packages
import numpy as np
from glob import glob
import os, shutil, sys
from copy import deepcopy
from termcolor import colored
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# local packages
from UNet import UNet
from ONet import ONet
from PSPNet import PSPNet
from data_train_generator import trainGenerator, get_image_info, Monitor_Training

# metrics and losses; anything new can be added here
from metrics_losses import dice, loss_dice, weighted_dice, loss_weighted_dice


class initialize_network(object): # The main class
    def __init__(self):
        ######################################################################## Initial
        ######################################################################## Values
        self.train_path = args["dataset"]
        self.saving_path = args["output"]

        self.mask_folder = "mask"
        self.image_folder = "image"

        self.image_format = args["image_format"]
        self.validation_path = args["validation_path"]

        self.num_epoch = args["num_epoch"]
        self.batch_size = args["batch_size"]
        self.num_class = args["num_classes"]
        self.save_period = args["steps_saved"]

        self.model = args["model_type"]
        self.backbone = args["backbone"]

        self.Activation = args["Activation"]
        self.image_final_size = args["image_final_size"]


        # print( "Checking if bach_size is higher than one or not. It should be more." )
        # assert self.batch_size>1


        ###### A couple of hard coded parameters which can be varied if needed
        # This is data augmentation; if needed, it can be modified
        self.aug_dict = dict(rotation_range=22.5,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='nearest')


        # adam optimizer parameters
        self.lr_init = 1e-4
        self.lr_decay = 5e-4

        # 8 bits images; Keras accepts just 8 bits images
        self.A_Range = 2**8-1


        # keeping track of your parameters
        self.FILE_txt = os.path.join("parameters.txt")
        with open(self.FILE_txt, "w") as f:
            for key, value in (self.__dict__).items():
                f.write('%s:%s\n' % (key, str(value)))
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)






    def train_network(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        print(colored("TRAINING IS STARTED; please wait ...", 'yellow'))

        # Moving the info for tracking your parameters
        shutil.move(self.FILE_txt, os.path.join(self.saving_path, self.FILE_txt))

        # Train generator for valdiation and training
        Train_set = trainGenerator(self, self.train_path, Mode="train")
        (self.target_size, self.image_dimension, self.num_train_image,
                self.image_color_mode) = get_image_info(self, self.train_path,
                                                        self.image_folder)

        # If validation exists
        if self.validation_path!="0":
            Val_set = trainGenerator(self, self.validation_path, Mode="validation")
            _, _, self.num_val_image, _ = get_image_info(self,
                                        self.validation_path, self.image_folder)


        # If image dimension provided use it
        if self.image_final_size>-1:
            self.target_size = (self.image_final_size, self.image_final_size)


        # Model checkpoint or any other callbacks should come here
        model_checkpoint = ModelCheckpoint(os.path.join(self.saving_path,
                                                "weights_M_{epoch:05d}.h5"),
                                                save_weights_only=True,
                                                period=self.save_period)

        JSON_PATH = os.path.sep.join([self.saving_path, "History.json"])

        callbacks = [Monitor_Training(JSON_PATH), model_checkpoint]



        # Generate models based on what has been asked by user; default is O-Net
        if self.model=="UNet":
            model = UNet(self.num_class, (self.target_size[0],
                                          self.target_size[1],
                                          self.image_dimension),
                         self.backbone,
                         self.Activation)

        elif self.model=="PSPNet":
            model = PSPNet(self.num_class, (self.target_size[0],
                                            self.target_size[1],
                                            self.image_dimension),
                           self.backbone,
                           elf.Activation)

        else:
            model = ONet(self.num_class, (self.target_size[0],
                                          self.target_size[1],
                                          self.image_dimension),
                         self.backbone,
                         self.Activation)


        # Compile
        model.compile(optimizer=Adam(lr=self.lr_init, decay=self.lr_decay),
                          loss=[loss_weighted_dice], metrics=[weighted_dice, dice])


        # fit the model with provided data
        print(self.validation_path)
        if self.validation_path!="0":
            model.fit_generator(Train_set,
                    validation_data=Val_set,
                    steps_per_epoch=np.ceil(self.num_train_image/self.batch_size),
                    validation_steps=np.ceil(self.num_val_image/self.batch_size),
                    epochs=self.num_epoch,
                    callbacks=callbacks)

        else:
            model.fit_generator(Train_set,
                    steps_per_epoch=np.ceil(self.num_train_image/self.batch_size),
                    epochs=self.num_epoch,
                    callbacks=callbacks)



        # Save the model
        MODEL_PATH = os.path.sep.join([self.saving_path, "Model.h5"])
        model.save(MODEL_PATH)

        print(colored("TRAINING IS DONE.", 'green'))




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Info = initialize_network()
    Info.train_network()
