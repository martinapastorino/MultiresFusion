
# CFC-CRF: Cluster level Fully Connected CRF

A Matlab-based implementation of the remote sensing image classification method proposed in "A Semisupervised CRF Model for CNN-Based Semantic Segmentation With Sparse Ground Truth", written by L. Maggiolo, D. Marcos, G. Moser, and S.B. Serpico, IEEE Trans. on Geoscience and Remote Sensing, 2021.
Please refer to this paper for a more detailed description of the algorithm.

A preliminary presentation of this work was published by the authors in a conference paper: L. Maggiolo, D. Marcos, G. Moser, and D. Tuia, “Improving maps from
cnns trained with sparse, scribbled ground truths using fully connected crfs,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), Valencia, Spain, Jul 2018, pp. 2099–2102.

##CFC-CRF Code
Quick overview of **CFC-CRF**'s code structure:

- **`ClusterLevel_FullyConnected_CRF.m`**: main script for running the proposed method. It requires precomputed multiscale tensor and corresponding clustering partition.
- **`PCA_Intermediate_Activations_and_Create_Multiscale_Tensor.m`**: script used to create the multiscale tensor. It takes image data and additional intermediate activations coming from CNN models as inputs. The script applies PCA to the intermediate activations, selects principal components and join them to the image data in order to create the multiscale tensor.
- **`Clusters_from_Tensor_Subset.m`**: Code which runs k-means on a random subset of the pixels of the multiscale tensor and save the clusetring partition that will be used in the method.
- **`CFC-CRF_Model_Source`**: Functions implementing the prosed CFC-CRF.
- **`Utils`**: Utilities functions used by the method.

##Data and Results
- data: contains the original image data of the two considered datasets (Vaihingen and Potsdsam) and the poseriors obatined from the CNN models. 
- Tensor-Clusers_data: location where the multiscale tensors and clustering partitions are stored. 
- results: directory where to store the results of CFC-CRF.

Note: A working example corresponding to Vaihingen in combination con U-net is available at this link: https://drive.google.com/drive/folders/1OCw7Oa6Mbj-SbvslDACXnZBUprQDYVsg?usp=sharing . 

####Run the scripts

To run the method on the available example, starting from image data and intermediate activations:

	run PCA_Intermediate_Activations_and_Create_Multiscale_Tensor
	run Clusters_from_Tensor_Subset
    run ClusterLevel_FullyConnected_CRF

The method is applied by diving the images in patches and the results are saved accordingly. This was done to allow for easier recovery but also to have simple ways of parallelizing computations. In order to rebuild the resulting classification map from the patches and compute the scores:

    run Assemb_Img_RemBord_Indexes

###Hyperparameters
CFC-CRF involves several hyperparameters, to be defined at diferent stages.

**`PCA_Intermediate_Activations_and_Create_Multiscale_Tensor.m`**: 
- L: number of considered intermediate activations blocks considered rom the CNN;
- p: number of principal components selected from each activation block and used to create the multiscale tensor.

**`Clusters_from_Tensor_Subset.m`**:
- `cluster_num_global`: defines the total number of clusters centroids k. Default value: 256;
- `cluster_neighb`: the number h of nearest centroids. Default value: 4.
- `unary_multiplier`: meant to reflect the confidence of the clustering partition, the higher tthe values the higher pennalization on changing the label of cluster. Deault value is #pixels_per_patch/k.

**`CFC-CRF_Model_Source`**
- `lambda.pp`: weight regulating the intensity of the graph connections from pixels to pixels;
- `lambda.cc`: weight regulating the intensity of the graph connections from cluster centroids to other cluster centroids (which are fully connected);
- `lambda.pc`: weight regulating the intensity of the graph connections from pixels to nearest cluster centroid (h-nn).

For guidelines about parameters setting, please refer to "A Semisupervised CRF Model for CNN-Based Semantic Segmentation With Sparse Ground Truth", written by L. Maggiolo, D. Marcos, G. Moser, and S.B. Serpico, IEEE Trans. on Geoscience and Remote Sensing, 2021.

####disclaimer
The present code was intended for laboratory experiments only.
