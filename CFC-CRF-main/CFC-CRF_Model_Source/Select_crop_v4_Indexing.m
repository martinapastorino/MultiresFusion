function [Features_Patch, Image_Patch, Tensor_Patch] = Select_crop_v4_Indexing(...
            matPrediction, Image_full, matTensor, Patch_division, Image_Patch)
            %features_full, x, gt, dsm, x_tot_global, xsy, xsx, offset_y, offset_x, scale_num, Pooling_index )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Cuts te atch to process, based on Patch_division

xsy = Patch_division.xsy;
xsx = Patch_division.xsx;
offset_y = Image_Patch.offset_y;
offset_x = Image_Patch.offset_x;

Features_Patch.prediction = matPrediction.prediction_noBorders(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);
                                           
Image_Patch.RGB = Image_full.RGB(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);
%Image_Patch.DSM = Image_full.DSM(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);
Image_Patch.Gt  = Image_full.Gt(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);

Tensor_Patch = matTensor.Multiscale_Tensor(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);
end

