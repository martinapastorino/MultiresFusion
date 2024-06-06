# Multiresolution fusion of hyperspectral and panchromatic images

This repository contains the code related to the multiresolution fusion of hyperspectral and panchromatic images for supervised classification tasks.  

## Installation

The code was built on a virtual environment running on Python 3.9

### Clone the repository

```
git clone --recursive https://github.com/martinapastorino/MultiresFusion.git
```

## Project structure

```
semantic_segmentation
├── dataset - contains the data loader
├── input - images to train and test the network 
├── net - contains the loss, the network, and the training and testing functions
├── CFC-CRF - contains the approximation of the ideal fully connected CRF (refer to the README in that folder for more informations)
├── output - should contain the results of the training / inference
|   ├── exp_name
|   └── model.pth
├── utils - misc functions
└── main.py - program to run
```
  
## Data

The model is trained and tested with PRISMA products, © of the Italian Space Agency (ASI).
## License

The code is released under the GPL-3.0-only license. See `LICENSE.md` for more details.

## Acknowledgements

Project carried out using PRISMA Products, © of the Italian Space Agency (ASI), delivered under a license to use by ASI. 

