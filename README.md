# Uncertainty-aware Image Caption Generation 

This repository contains the implementation of paper: Uncertainty-aware Image Captioning. 


### Our model consists of two module: 

1. Uncertainty Measurement: image conditioned bag-of-words 

input: image regions, output: distribution of word vocabulary 

2. Caption Generation: uncertainty-aware image caption 

input: image region, last stage sequence, out put: next stage sequence  


### The main contirbutions: 

1. We propose a new uncertainty-aware model for image caption generation. Compared with previous work, our model allow difficulty control over generation and enjoy a siginificant reduction over emperical time complexity. 

2. We introduce a cross-modality uncertainty estimation model inspired on the idea of bag-of-word. Based on the token-level uncertainty estimation, a recursive algorithm is applied to contruct the training set. 

3. We devdelop a uncertainty-adopted beam search algorithm to improve the decoding effeciency.  

4. Experiments on MS COCO benchmark demonstrate the superiority of our UAIC model over strong baselines. 


### Data Preparation: 


To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.

Detection features are computed with the code provided by [1]. To reproduce our result, please download the COCO features file [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx) (~53.5 GB), in which detections of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`), and each value should be a `(N, 2048)` tensor, where `N` is the number of detections. 
 


### Model Training: 

1. Uncertainty Measurement: image conditioned bag-of-words 

2. Caption Generation: uncertainty-aware image caption  




