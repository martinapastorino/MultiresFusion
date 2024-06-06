clear
%Run clustering on a random subset of the Multiscale Tensor
folder = fileparts(which('Clusters_from_Tensor_Subset')); 
addpath(genpath(strcat(folder,'\Utils')));

dataset = 'Vaihingen';  %dataset is either 'Vaihingen' or 'Potsdam'
CNN_model = 'Unet';     %CNN_model is either 'Hypercolumn' or 'Unet'
cur_dir = pwd;

%Example data only for the case of Vaihingen dataset with Unet model

if strcmp(dataset,'Potsdam')
    dir = strcat(folder,'\data\Potsdam\');
    rowIdx = [6 6 6 7 7 7];
    colIdx = [7 8 9 7 8 9];	
    if strcmp(CNN_model,'Hypercolumn')
        fprintf('No example available for this combination.\n')
    elseif strcmp(CNN_model,'Unet')
        fprintf('No example available for this combination.\n')
    else
        fprintf('Non valid CNN model.\nCNN model can be either Unet or Hyercolumn.\n')
    end
elseif strcmp(dataset,'Vaihingen')
    rowIdx = [11 15 28 30];
    dir = strcat(folder,'\data\Vaihingen\');
    if strcmp(CNN_model,'Hypercolumn')
        fprintf('No example available for this combination.\n')
    elseif strcmp(CNN_model,'Unet')
        rng('shuffle');
        num = 4;
        img_num = num2str(rowIdx(num));
        
        Param.Clust = struct('use_clusters', true, 'cluster_num_global', 256, 'rand_clustering', true,...
                             'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.02,...
                             'unary_multiplier', 2800, 'c_scale_num', []);

        data_dir = 'Tensors-Clusters_data\Vai_Unet_Tensors';
        cd(data_dir);
        matTensor = matfile(strcat('Multiscale_Tensor_Unet_',img_num,'_ScribGt.mat'));
        matPrediction = matfile(strcat('posteriors_img_',img_num,'.mat'));
        cd(cur_dir);
        %Use this part to create new kinds of tensors
        Multiscale_Tensor = matTensor.Multiscale_Tensor;

        dir = strcat(folder, '\data\Vaihingen\');
        post_dir = 'outputs\Unet_f2_p2\';
        name = strcat(post_dir,'top_mosaic_09cm_area',img_num,'_posteriors.tif');
        posteriors_tif = Tiff(strcat(dir,name),'r');
        posteriors = read(posteriors_tif);
        close(posteriors_tif);
        features_full.prediction = matPrediction.posteriors;

        fprintf('computing clusters...\n');
        [Cluster_Data.Centroids, Cluster_Data.Posterior_Prob, Cluster_Data.Variance] = ...
            Generate_Clusters_Unary_Grid_New( Multiscale_Tensor, features_full.prediction,...
                    Param.Clust.cluster_num_global, Param.Clust.rand_clustering, 20 );

        save_clust = true;
        if save_clust
            Clusters = struct('Centroids',Cluster_Data.Centroids,'Posterior_Prob',Cluster_Data.Posterior_Prob,...
                'Variance', Cluster_Data.Variance);
            name_cl = strcat('Clusters_img_',img_num,...
                             sprintf('_scrib_tensor_Unet_%icl',...
                                     Param.Clust.cluster_num_global));
            cd(data_dir);
            if isfile(strcat(name_cl,'.mat'))
                name_cl_new = strcat(name_cl,'_new');
                save(name_cl_new,'Clusters','-v7.3');
            else
                save(name_cl,'Clusters','-v7.3');
            end
            cd(cur_dir);
        end 
    else
        fprintf('Non valid CNN model.\nCNN model can be either Unet or Hyercolumn.\n')
    end
else
    fprintf('Non valid dataset.\nDataset can be either Vaihingen or Potsdam.\n')
end