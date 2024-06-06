% Attempt at clustering
data_dir = 'C:\Users\mpastori\OneDrive - unige.it\Documents\PhD\Projects\PRISMA_Learn\DATA\2018IEEE_Contest\Phase2\FullHSIDataset\';
% tens_name = strcat(data_dir,'multiscale_tensor_pca_8comp.tif');
% tens_tif = Tiff(tens_name,'r');
% tensor = read(tens_tif);
% close(tens_tif);
% tensor = double(tensor);
% save(strcat('Multiscale_Tensor_UNet.mat'),'tensor','-v7.3');

post_name = strcat(data_dir,'posteriors.tif');
post_tif = Tiff(post_name,'r');
posteriors = read(post_tif);
close(post_tif);
posteriors = double(posteriors);
save(strcat('posteriors.mat'),'posteriors','-v7.3');


%% Section with attempt at clustering

Param.Clust = struct('use_clusters', true, 'cluster_num_global', 256, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.2,...
                     'unary_multiplier', 2800, 'c_scale_num', []);

matTensor = matfile(strcat('Multiscale_Tensor_UNet.mat'));
matPrediction = matfile(strcat('posteriors.mat'));
% cd(cur_dir);
%Use this part to create new kinds of tensors
Multiscale_Tensor = matTensor.tensor;

features_full.prediction = matPrediction.posteriors;
addpath('./Utils/');
fprintf('computing clusters...\n');

% [Cluster_Data.Centroids, Cluster_Data.Posterior_Prob, Cluster_Data.Variance] = ...
%     Generate_Clusters_Unary_Grid_New( Multiscale_Tensor, features_full.prediction,...
%             Param.Clust.cluster_num_global, Param.Clust.rand_clustering, 30);

% function [ C, Cluster_unary, Cluster_variance ] = Generate_Clusters_Unary_Grid_New( x_tot, act, cluster_num, rand_grid, pix_step)
%Generate_Clusters_Unary_Grid_New Generates clusters from a tensor (image)
%using a Kmeans algorithm
%   Detailed explanation goes here

%should increase up to at least 1% of total pixels
%check for grid subsampling
pix_step = 30;
x_tot = Multiscale_Tensor;
noise_range = pix_step-1;

%generate sampling grid (with noise)
limit_x = size(x_tot,2)-pix_step;    limit_y = size(x_tot,1)-pix_step;
x = 1:pix_step:limit_x;
y = 1:pix_step:limit_y;
[X,Y] = meshgrid(x,y);
noise_x = randi(noise_range,size(y,2),size(x,2));
x_noisy = X + noise_x;
noise_y = randi(noise_range,size(y,2),size(x,2));
y_noisy = Y + noise_y;

%% continuation

x_tot_r = reshape(x_tot, size(x_tot,1)*size(x_tot,2),size(x_tot,3));
linearInd = sub2ind(size(x_tot), y_noisy, x_noisy);

linearInd_r = reshape(linearInd, size(linearInd,1)*size(linearInd,2),1);
select_tot = x_tot_r(linearInd_r,:);

%spatial_w = 0.07;
border = 8;
act = features_full.prediction;
[~,label] = max(act,[],3);
CLASS = reshape(label,1,size(label,1)*size(label,2));
actRow = reshape(act,size(act,1)*size(act,2),size(act,3))'; 
x_row = double(reshape(x_tot,size(x_tot,1)*size(x_tot,2),size(x_tot,3)));
 
%% running kmeans

fprintf('running kmeans...\n');
%[~,C] = kmeans(select_tot,cluster_num);
options_km = statset('UseParallel',1);
cluster_num = 256;
[~,C] = kmeans(select_tot,cluster_num,'MaxIter',500,'Options',options_km,'Replicates',8);
%'MaxIter',1

fprintf('assigning unarys...\n');
yf = size(C,1);
class_num = size(act,3);
Cluster_unary = (1/class_num)*ones(class_num,yf);
Cluster_variance = (1/class_num)*ones(class_num,yf);

%% final part

[~,idx] = pdist2(C,x_row, 'euclidean','Smallest',1);
for i=1:size(C,1) %for each cluster
    belong_to_clust = find(idx==i);
    Cluster_unary(:,i) = (sum(actRow(:,belong_to_clust),2))/nnz(belong_to_clust);
    for c=1:class_num %6
        Cluster_variance(c,i) = var(actRow(c,belong_to_clust));
    end
end
% %Cluster_unary = -log(Cluster_unary + eps);
% 
% end

     
% % x_tot = Multiscale_Tensor; % act = features_full.prediction;
% % cluster_num = Param.Clust.cluster_num_global; % rand_grid = Param.Clust.rand_clustering;
% 
% save_clust = true;
% if save_clust
%     Clusters = struct('Centroids',Cluster_Data.Centroids,'Posterior_Prob',Cluster_Data.Posterior_Prob,...
%         'Variance', Cluster_Data.Variance);
%     name_cl = strcat('Clusters_img_',img_num,...
%                      sprintf('_scrib_tensor_Hyperc_%icl',...
%                              Param.Clust.cluster_num_global));
%     cd(data_dir);
%     if isfile(name_cl)
%         name_cl_new = strcat(name_cl,'_new');
%         save(name_cl_new,'Clusters','-v7.3');
%     else
%         save(name_cl,'Clusters','-v7.3');
%     end
%     cd(cur_dir);
% end 