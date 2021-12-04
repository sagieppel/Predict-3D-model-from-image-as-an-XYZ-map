# Run net on folder of images and display results

import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import Visuallization as vis
import cv2
import open3d as o3d
import RGBD_To_XYZMAP
#------------------input parameters-------------------------------------------------------------------------------
InputImage=r"Example/Test.jpg" # Input image file
Trained_model_path =  "logs/Defult.torch" # Train model to use

DisplayXYZPointCloud=True # Show Point cloud
MaxSize=900
#************************************Masks and XYZ maps to predict********************************************************************************************************
#******************************Create and Load neural net**********************************************************************************************************************

Net=NET_FCN.Net() # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # choose gpu or cpu
Net=Net.to(device)
Net.eval()
#*********************************Read image and resize*******************************************************************************

Img=cv2.imread(InputImage)
Img=vis.ResizeToMaxSize(Img,MaxSize)

ImBatch=np.expand_dims(Img,axis=0)
###############################Run Net and make prediction###########################################################################
with torch.no_grad():
    XYZMap = Net.forward(Images=ImBatch,TrainMode=False) # Run net inference and get prediction

#DepthMap=LogDepthMap
#----------------------------Convert Prediction to numpy-------------------------------------------
XYZMap=XYZMap.transpose(1,2).transpose(2, 3).data.cpu().numpy()[0]
#-----------------------------Convert Depth to XYZ map-------------------------------------------------------------------

#DepthMapReal= cv2.imread("Example/Depth2.png",-1)
#******************************

vis.show(Img,"Image "+InputImage)
ROI=XYZMap[:,:,0]*0+1
vis.ShowXYZMap(XYZMap, "XYZ Map")
vis.DisplayPointCloud(XYZMap,Img,ROI,step=2)
