clear
folder = fileparts(which('PRISMA_fully_connected_CRF')); 
addpath(genpath(strcat(folder,'\CFC-CRF_Model_Source')));
addpath(genpath(strcat(folder,'\Utils')));

%dataset = 'Vaihingen';  %dataset is either 'Vaihingen' or 'Potsdam'
dataset = 'PRISMA';
CNN_model = 'FCN_SS';     %CNN_model is either 'Hypercolumn' or 'Unet'
%Example data only for the case of Vaihingen dataset with Unet model

%Hyperparameters configuration
Param = struct('Save',[],'Tensor',[],'Clust',[],'Load',[]);
Param.Save = struct('save_data', true, 'model', [], 'gtType', []);
if ~Param.Save.save_data
    sprintf('will not save results')
end
Param.Tensor = struct('useRGB', true, 'useDsm', true, 'normalize', true, ...
                      'spatial_weight', 0.05, 'preprocessing', false,  'band_weight', [0.9,1,1,0.8]);
                 
Param.Clust = struct('use_clusters', true, 'cluster_num_global', 64, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.05,...
                     'unary_multiplier', 2800, 'c_scale_num', []);
                 
Param.Load = struct('scale_num', 1, 'concat', true, 'feat_x', true, 'eroded', true);
Param.Clust.c_scale_num = Param.Load.scale_num;
cur_dir = pwd;

%Load image data: optical channels, ground truth, digital surface model
if strcmp(dataset,'PRISMA')
    num = 1;
    dir = strcat(folder,'\data\PRISMA\ER');
    rowIdx = ["PRS_L2D_STD_20220715102100_20220715102104_0001"];
    %rowIdx = ["PRS_L2D_STD_20210402103032_20210402103036_0001", "PRS_L2D_STD_20210831103404_20210831103409_0001"];
    img_num = num2str(rowIdx(num));
    [Image_full.RGB,~] = imread(strcat(dir,'/',img_num,'/',img_num,'_PAN.tif'));
    [Image_full.Gt,~] = imread(strcat(dir,'/',img_num,'/',img_num,'_gt.png'));

    %[Image_full.DSM,~] = imread(strcat(data_dir,'/ndsm/dsm_09cm_matching_area',img_num,'_normalized.jpg'));
    clear data_dir;
else
    fprintf('Non valid dataset.\nDataset can be either PRISMA or PRISMA.\n')
    return
end

if strcmp(CNN_model,'FCN_SS')
    if strcmp(dataset,'PRISMA')
        data_dir = 'PRISMA_Tensors/ER/FCN_SS/';
        cd(data_dir);
        matPred = matfile(strcat('posteriors_',img_num,'.mat'));
        matPrediction.prediction_noBorders = matPred.posteriors;
        matTensor = matfile(strcat('Multiscale_Tensor_',img_num,'_ScribGt.mat'));
        cd(cur_dir);
    else 
        fprintf('Non valid dataset. \nDataset can only be PRISMA. Look at the other script!!!')
    end
else
    fprintf('Non valid CNN model.\nCNN model can be either NN or NN.\n')
    return
end

fprintf('Select clusters file for %s image %s\n', dataset, img_num)
cd(data_dir);
if exist('Clusters','var')==false
    [clust_file_name,path] = uigetfile('*.mat','select a mat file');
    clust_file_name = fullfile(path,clust_file_name);
    clear path
    name_clust_match = strfind(clust_file_name,img_num);
    if ~isempty(name_clust_match)
        load(clust_file_name);
    end
end
Cluster_Data = Clusters; clearvars Clusters
cd(cur_dir);
Cluster_Data.Centroids = cast(Cluster_Data.Centroids,'double');

sy = size(matTensor.Multiscale_Tensor,1);
sx = size(matTensor.Multiscale_Tensor,2);
Image_full.RGB = double(Image_full.RGB(1:sy,1:sx,:));
Image_full.Gt = Image_full.Gt(1:sy,1:sx,:);
%Image_full.DSM = Image_full.DSM(1:sy,1:sx,:);
fprintf('Data loaded correctly.\n')

%%
elapsed_time = zeros(144,1);
%Set the criteria for diving the image in patches
img_size_y = sy; img_size_x = sx;
Patch_division = struct('xsy',600,'xsx',600, 'border',50, 'indy',[],'indx',[]);
[Patch_division.indy, Patch_division.indx ] = Patches_Starting_Points_Pots_v3...
      (img_size_y, img_size_x, Patch_division.xsx, Patch_division.border );
  
Image_Patch = struct('offset_x',[],'offset_y',[],'RGB',[],'Gt',[],'DSM',[]);

%% Run Cl-FC-CRF on each patch separately (the process can be parallelized)
for r=1:size(Patch_division.indy,1)
Image_Patch.offset_y = Patch_division.indy(r,1);

for c=1:size(Patch_division.indx,1)   %c=1:7:8
    Image_Patch.offset_x = Patch_division.indx(c,1);
    fprintf('patch_row_%i_col_%i...\n',Image_Patch.offset_y,Image_Patch.offset_x);
    
    %Select image patch
    [Features_Patch, Image_Patch, Tensor_Patch] = Select_crop_v4_Indexing(matPrediction,...
        Image_full, matTensor, Patch_division, Image_Patch);
    Tensor_Patch = cast(Tensor_Patch, 'double');
    lambda = struct('pp', 2, 'cc', 1, 'pc', 1);
    fprintf('lambda pixel-pixel = %d; lambda cluster-cluster = %d; lambda pixel-cluster = %d\n', lambda.pp, lambda.cc, lambda.pc);
    
    %Define the graph for the cannonical CRF
    [CLASS_tot_global, UNARY_tot_global, Pairw_tot_global, LABELCOST, sigma, Image_Patch, diff_tot] = ...
            Canonical_CRF(Param, Image_Patch, Features_Patch, lambda);
    
    %Add cluster level connection
    fprintf('cluster model...\n');
    [Pairw_tot_global, CLASS_tot_global, UNARY_tot_global, conn, Cluster_Data] = ...
        Cluster_Connected_Model_v2(Pairw_tot_global, CLASS_tot_global, UNARY_tot_global, ...
                                Cluster_Data, Tensor_Patch, sigma, lambda,...
                                Param.Clust);
    UNARY_tot_global = cast(UNARY_tot_global,'single');
    tic;
    fprintf('running graph cut...\n');
    curDir = pwd;
    %cd('..\MATLAB\Cl-FC-CRF_Scrib_GT_Vaihingen\gcmex-2.3.0\GCMex');
    cd('gcmex-2.3.0\GCMex');
    %[LABELS_global, ENERGY, ENERGYAFTER] = ...
    [LABELS_global,~,~] = GCMex(CLASS_tot_global-1, UNARY_tot_global, Pairw_tot_global, LABELCOST);
    cd(curDir);
    clear curDir
    t_gc_global=toc;
    elapsed_time(12*(r-1)+c,1) = toc;
    fprintf('graph cut with global clusters done...y=%i , x=%i...\n',Image_Patch.offset_y,Image_Patch.offset_x);
    
    [Image_Patch, Cluster_Data, new_img_global] = Get_GC_Results(conn, LABELS_global, ...
        Cluster_Data, Tensor_Patch, Image_Patch, Features_Patch, Patch_division, Param);
    
    if Param.Save.save_data
        Param.Save.model = 'Clust';
        Param.Save.gtType = strcat('fullGt'*(Param.Load.eroded==false), 'scrib'*Param.Load.eroded);

        Save_folder = strcat('results\ER\',CNN_model,'_',img_num,'_',Param.Save.model,'_',Param.Save.gtType,...
                             sprintf('_%icl',Param.Clust.cluster_num_global),...
                             sprintf('_L%i_%i_%i',lambda.pp,uint8(lambda.cc),uint8(lambda.pc)),... 
                             '_0,9_1_1_0,8',...
                             sprintf('_%in',Param.Clust.cluster_neighb),...
                             sprintf('_%iu',Param.Clust.unary_multiplier));
        if ~isfolder(Save_folder)
            mkdir(Save_folder);
        end

        %new_img_global = new_img_global{1};
        name = strcat('img_',img_num,'_',Param.Save.model,'_',Param.Save.gtType,...
                      sprintf('_y%i_x%i',Image_Patch.offset_y,Image_Patch.offset_x),...
                      sprintf('_%icl',Param.Clust.cluster_num_global),...
                      sprintf('_L%i_%i_%i',lambda.pp,uint8(lambda.cc),uint8(lambda.pc)),...
                      '_0,9_1_1_0,8',...
                      sprintf('_%in',Param.Clust.cluster_neighb),...
                      sprintf('_%iu',Param.Clust.unary_multiplier));
        curDir = pwd;
        cd(Save_folder);
        %new_img_global = Image_Patch.Graph_pred;
        save(name,'new_img_global');
        %save(name,'Image_Patch.Graph_pred');
        cd(curDir)
        clear name;
        clear curDir;
    end
    time = toc;
end
end
clear

curDir = pwd;
cd(Save_folder);
elaps_time_name = strcat('elapsed_time_img_',img_num,'_',Param.Save.model,'_',Param.Save.gtType,...
              sprintf('_%icl',Param.Clust.cluster_num_global),...
              sprintf('_L%i_%i_%i',lambda.pp,uint8(lambda.cc),uint8(lambda.pc)),...
              '_0,9_1_1_0,8',...
              sprintf('_%in',Param.Clust.cluster_neighb),...
              sprintf('_%iu',Param.Clust.unary_multiplier));
save(elaps_time_name,'elapsed_time');
cd(curDir)
