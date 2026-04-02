# Carbonate Texture Segmentation using U-Net and Attention U-Net

This repository contains the MATLAB implementation developed for a senior research project on semantic segmentation of petrographic textures related to carbonate rock imaging.

## Overview
The project investigates the performance of U-Net and Attention U-Net architectures for multi-class texture segmentation. The workflow is based on an adapted Randen-style benchmark, where texture images are used to construct a composite image with a known ground-truth mask.

## Methodology
- Nine grayscale petrographic texture images are used as segmentation classes
- A composite image is constructed using a predefined mask
- Training data is generated as 32×32 labeled patches
- Models are trained using different:
  - optimizers (SGDM, Adam, RMSprop)
  - epoch numbers
  - network depths
- Evaluation is performed on the composite image using pixel-wise accuracy and additional metrics (precision, recall, F1-score)

## Code Structure
### Data Preparation
- `changetrainRanden1.m` – replaces benchmark textures with petrographic textures
- `changeDataRanden1.m` – constructs the composite image
- `prepareTrainingLabelsRanden1.m` – generates training patches and labels

### U-Net
- `segmentationTextureUnet1.m` – main U-Net experiments
- `segmentationTextureUnet1_best.m` – best U-Net configuration (timing + metrics)

### Attention U-Net
- `segmentationTextureAttentionUnet1.m` – main Attention U-Net experiments
- `segmentationTextureAttentionUnet1_best.m` – best Attention U-Net configuration

### Attention U-Net Architecture
- `addAttentionGate.m`
- `attentionUnet15layers.m`
- `attentionUnet20layers.m`

## Sample Data
This repository includes:
- 9 petrographic texture images (`1.jpg` to `9.jpg`)
- Composite image and mask used for evaluation

## Results
The project compares U-Net and Attention U-Net in terms of:
- segmentation accuracy
- training time
- inference time
- precision, recall, and F1-score

## Benchmark Basis
The workflow is adapted from the Randen-style texture segmentation benchmark described in:

Karabag, C., Verhoeven, J., Miller, N. R., & Reyes-Aldasoro, C. C.  
"Texture Segmentation: An Objective Comparison between Five Traditional Algorithms and a Deep-Learning U-Net Architecture."  
Applied Sciences, 9(18), 3900, 2019.

## Requirements
- MATLAB (tested on R2024a)
- Deep Learning Toolbox
- Image Processing Toolbox

## Author
Lina Taha  
Khalifa University
