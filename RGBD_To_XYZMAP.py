import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import Visuallization as vis

def RGBD2XYZ(rgbd_image,K):
    M = K.intrinsic_matrix
    DepthMap = np.asarray(rgbd_image.depth)
    Img = np.asarray(rgbd_image.color)
    height = DepthMap.shape[0]
    width = DepthMap.shape[1]
    GridY = np.array(list(range(height))) - height/2
    GridY = np.transpose(np.tile(GridY, (width, 1)))
    GridX = np.array(list(range(width))) - width/2#(list(range(width)) - width/2)  # Might be +shift x https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    GridX = np.tile(GridX, (height, 1))
    XYZ = np.zeros([height, width, 3], dtype=np.float32)
    XYZ[:, :, 2] = DepthMap
    XYZ[:, :, 1] = DepthMap * GridY / M[0,0]
    XYZ[:, :, 0] = DepthMap * GridX / M[1,1]
    ROI = (DepthMap > 0).astype(np.float32)
    return XYZ,ROI,Img,DepthMap