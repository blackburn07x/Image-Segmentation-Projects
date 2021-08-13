# U-Net Implementation in Tensorflow 1.15
*on Carvana Image Segmentation Dataset on Kaggle*

***from the paper***: [https://arxiv.org/abs/1505.04597]

In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

This is a simple implementaiton of the model mentioned in the paper. 

**Dataset can be downloaded from here:** https://www.kaggle.com/c/carvana-image-masking-challenge

### About the implementation

The dataset masks are in RLE format. So firest this rle format is converted into image mask of numpy array and then used ofr training. 
- preprocess_data.py : This .py file pre-processes the RLE(.csv) file and produces numpy array of image mask. This file also pre-processes the input image into numpy array.
- utils.py : This .py file contains functions that are used to create U-net network.
- network.py : Actual implementation of U-net architecture.
- main.py : This file is used for training the U-net on Carvana dataset and saving the trained model and prediction into appropriate format

### Result

![alt text](output/full_figure%20(2).png)

### Conclusion

This project is on a simple implementation of U-net on a Binary Segmentation problem. However, with time I seek to improve the complexity of this project from Binary to Categorical and with a better implementation of the project.
