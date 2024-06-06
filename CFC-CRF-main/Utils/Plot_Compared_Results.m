function [ init_accuracy, new_accuracy ] = Plot_Compared_Results( init_pred, new_pred, gt )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%evaluate accuracy
pixel_num = numel(init_pred(:,:,1));
idy = all((new_pred == gt),3);    % R==G==B
new_accuracy = 1-double((pixel_num - sum(idy(:)))/(pixel_num));
idy = all((init_pred== gt),3);     % R==G==B
init_accuracy = 1-double((pixel_num - sum(idy(:)))/(pixel_num));

%plot results
figure; 
h    = [];
h(1)=subplot(1,3,1);
imagesc(init_pred, 'parent', h(1));
title(strcat(num2str(init_accuracy*100),'%'));
h(2)=subplot(1,3,2);
imagesc(new_pred, 'parent', h(2));
title(strcat(num2str(new_accuracy*100),'%'));
h(3)=subplot(1,3,3);
%imagesc(abs(init_pred - new_pred), 'parent', h(3));
imagesc(gt, 'parent', h(3));

idy_c = all((new_pred == init_pred),3);
difference = double((pixel_num - sum(idy_c(:)))/(pixel_num));
title(strcat(num2str(difference*100),'%'));
% title(strcat(num2str(...
% (nnz(abs(init_pred - new_pred))/(numel(new_pred(:,:,1))))*100),'%'));


end

