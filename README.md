# Overall Net (O-Net)


## Dependency:
To install the required packages please use file "requirment.txt".
To train the one of the networks presented in CVPR 2020, you can modify or use "train.py".
The codes are tested on systems with Python3 and for macOS (Catalina) and Linux OS (Ubuntu 18.04).



## Citing this work:
Please cite our paper in CVPR, if you use any codes or materials from this repository.



## Included files:
1) data_train_generator.py: Needed functions to load images and provide information from training.
2) encoders.py: ResNet, Simple, and VGG encoders.
3) metrics_losses.py: Any metrics or losses should go here.
4) ONet.py: Our decoder.
5) PSPNet.py: PSP Net decoder.
6) UNet.py: U-Net Net decoder.
7) train.py: The main code to combine all functions and modules to train networks and save results.
8) Images: 50 sample images for training and 10 sample images for validation (this will be increased or linked to full dataset on GitHub).



## How to use the code:
File "train.py" shows an example to compile and use models.
A help has been included in the "train.py" code which can be used to check more detailed description of variables, like number of epochs and etc.
This can be seen as follow:
python3 PATH_TO_PACKAGE/train.py -h

An example of training networks with validation path:
python3 PATH_TO_PACKAGE/train.py -i PATH_TO_PACKAGE/Images/train -o ~/Desktop/UNET -v PATH_TO_PACKAGE/Images/validation -ne 100 -bs 3 -ifs 256 -nc 4 -mt UNet

An example of training networks without validation path:
python3 PATH_TO_PACKAGE/train.py -i PATH_TO_PACKAGE/Images/train -o ~/Desktop/UNET -ne 100 -bs 3 -ifs 256 -nc 4 -mt UNet



## Notes on how to use the code:
1) If "-ifs" in "train.py" is not given, it will assume the input image size is correct and ready for training. If it is given, then, it resize the image to the requested resolution.
2) The masks should be in gray-scale. Each intensity level shows one class for classification. For example, the sample dataset has four classes which generates 0, 1, 2, and 3 for intensity levels corresponding to the classes of segmentation.
3) Any preprocessing steps can be added in "data_train_generator.py" as mentioned in the file.
4) Any metrics can be added in the "metrics_losses.py".



## Important Notes about the small dataset:
1) This dataset has three subfolders for "train" and "validation" sets: "image", "mask", and "mask_visualization". The "mask_visualization" is just for visualizing the masked images as four masks maximum intensity is 4 and hard to be differentiated.
2) This is small dataset just for proving that the networks working; the performance of the networks are low on validation set and high on training set.


## Description of files in output of "train.py":
By running "train.py", the following files are generated:
1) "Model.h5": The model architecture is saved using this file.
2) "weights_M_00002.h5": It contains weights for epoch 00002
3) "History.json": It keeps the history of training, including training and validations for each epoch
