%Compute PCA on intermediate Activations and build the Multiscale Tensor
folder = fileparts(which('PRISMA_fully_connected_CRF')); 

%dataset is either 'Vaihingen' or 'Potsdam'
dataset = 'PRISMA'; 
%CNN_model is either 'Hypercolumn' or 'Unet'
CNN_model = 'FCN_SS';
cur_dir = pwd;
save_dir = 'PRISMA_Tensors/ER/FCN_SS/';


%% Import Multiscale Tensor
if strcmp(dataset,'PRISMA')
	%Given the big size of the images, PCA are precomputed and saved before creating the multiscale tensor
	data_dir = strcat(folder,'\data\PRISMA\ER');
    dir = data_dir;
    %rowIdx = ["PRS_L2D_STD_20210402103032_20210402103036_0001", "PRS_L2D_STD_20210831103404_20210831103409_0001"];
    rowIdx = ["PRS_L2D_STD_20220715102100_20220715102104_0001"];
    
    % import Multiscale tensor and posterior tensor images computed in Python (see other script to compute
    % PCA)
    for num = 1
        img_num = rowIdx(num);
        %Multiscale_Tensor_tif = Tiff(strcat(dir,'\',img_num,'\', img_num, '_Multiscale_Tensor_10comp.tif'), 'r');
        Multiscale_Tensor_name = strcat(dir,'\',img_num,'\',CNN_model,'\', img_num, '_Multiscale_Tensor_7comp.tif');
        posteriors_name = strcat(dir,'\',img_num,'\', CNN_model,'\', img_num, '_posteriors.tif');
        posteriors_tif = Tiff(strcat(dir,'\',img_num,'\', CNN_model,'\', img_num, '_posteriors.tif'), 'r');
        %Multiscale_Tensor = read(Multiscale_Tensor_tif);
        Multiscale_Tensor = imread(Multiscale_Tensor_name);
        %posteriors = read(posteriors_tif);
        posteriors = imread(posteriors_name);

        cd(save_dir)
        save(strcat('Multiscale_Tensor_',img_num,'_ScribGt.mat'),'Multiscale_Tensor','-v7.3');
        save(strcat('posteriors_',img_num,'.mat'),'posteriors','-v7.3');
        cd(cur_dir)

    end
end

%     figure; imshow(Multiscale_Tensor(:,:,1));
%     figure; imshow(Multiscale_Tensor(:,:,2:7));
%     figure; imshow(Multiscale_Tensor(:,:,8:10));
    
%% Run Clustering

Param.Clust = struct('use_clusters', true, 'cluster_num_global', 64, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.5,...
                     'unary_multiplier', 2800, 'c_scale_num', []);

cur_dir = pwd;
num = 1;
img_num = rowIdx(num);

cd(save_dir)
matTensor = matfile(strcat('Multiscale_Tensor_',img_num,'_ScribGt.mat'));
matPrediction = matfile(strcat('posteriors_',img_num,'.mat'));
cd(cur_dir);
data_dir = 'PRISMA_Tensors/ER/FCN_SS/';
%Use this part to create new kinds of tensors
Multiscale_Tensor = matTensor.Multiscale_Tensor;

features_full.prediction = matPrediction.posteriors;
addpath('./Utils/');
fprintf('computing clusters...\n');
[Cluster_Data.Centroids, Cluster_Data.Posterior_Prob, Cluster_Data.Variance] = ...
    Generate_Clusters_Unary_Grid_New( Multiscale_Tensor, posteriors,...
            Param.Clust.cluster_num_global, Param.Clust.rand_clustering, 100);
     
% x_tot = Multiscale_Tensor; % act = features_full.prediction;
% cluster_num = Param.Clust.cluster_num_global; % rand_grid = Param.Clust.rand_clustering;

save_clust = true;
if save_clust
    Clusters = struct('Centroids',Cluster_Data.Centroids,'Posterior_Prob',Cluster_Data.Posterior_Prob,...
        'Variance', Cluster_Data.Variance);
    name_cl = strcat('Clusters_img_',img_num,...
                     sprintf('_scrib_tensor_%icl',...
                             Param.Clust.cluster_num_global));
    cd(data_dir);
    if isfile(name_cl)
        name_cl_new = strcat(name_cl,'_new');
        save(name_cl_new,'Clusters','-v7.3');
    else
        save(name_cl,'Clusters','-v7.3');
    end
    cd(cur_dir);
end 