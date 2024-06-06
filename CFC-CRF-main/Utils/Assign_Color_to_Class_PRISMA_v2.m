function [ Class_RGB ] = Assign_Color_to_Class_PRISMA_v2( label )
%Assign_Color_to_Class
%   This function assigns colors to class in base of a predefined colormap
%   (this apply only for 6 possible classes)
Class_RGB = label;
Class_RGB(:,:,1) = 144*(label==1) + 213*(label==2) + 241*(label==3) + 21*(label==4) + 180*(label==5) + 251*(label==6) + 31*(label==7);
Class_RGB(:,:,2) = 7*(label==1) + 213*(label==2) + 238*(label==3) + 75*(label==4) + 255*(label==5) + 154*(label==6) +  35*(label==7);
Class_RGB(:,:,3) = 48*(label==1) + 213*(label==2) + 79*(label==3) + 29*(label==4) + 146*(label==5) + 153*(label==6) +  180*(label==7);

end

%colormap

% palette = {0 : (144, 7, 48),   # built-up OK
%            1 : (213, 213, 213),     # streets OK
%            2 : (241, 238, 79),     # crop soil OK
%            3 : (21, 75, 29), # trees OK
%            4 : (180, 255, 146), # grass OK
%            5 : (251, 154, 153), # bare soil
%            6 : (31, 35, 180), # water OK
%            7 : (0, 0, 0)} # unlabeled OK
% inverse tranform from RGB groundtruth to class labels
%gt = gt/255;
% class_gt(:,:) = 1*(((gt(:,:,1)==1)+(gt(:,:,2)==1)+(gt(:,:,3)==1))==3)+...
%                 2*(((gt(:,:,1)==0)+(gt(:,:,2)==0)+(gt(:,:,3)==1))==3)+...
%                 3*(((gt(:,:,1)==0)+(gt(:,:,2)==1)+(gt(:,:,3)==1))==3)+...
%                 4*(((gt(:,:,1)==0)+(gt(:,:,2)==1)+(gt(:,:,3)==0))==3)+...
%                 5*(((gt(:,:,1)==1)+(gt(:,:,2)==1)+(gt(:,:,3)==0))==3)+...
%                 6*(((gt(:,:,1)==1)+(gt(:,:,2)==0)+(gt(:,:,3)==0))==3);

