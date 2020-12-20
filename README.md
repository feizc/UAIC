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









