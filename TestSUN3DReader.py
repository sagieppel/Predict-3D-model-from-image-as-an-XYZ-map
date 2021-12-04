#http://www.open3d.org/docs/0.8.0/tutorial/Basic/rgbd_images/redwood.html
# http://redwood-data.org/indoor/dataset.html
# examples/Python/Basic/rgbd_redwood.py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import RGBD_To_XYZMAP
import Visuallization as vis
import SUN3DReader as DepthReader

MinSize=300 # Min image dimension (height or width)
MaxSize=800# Max image dimension (height or width)
MaxPixels=800*800*2# Max pixels in a batch (not in image), reduce to solve out if memory problems
MaxBatchSize=6# Max images in batch

DataDir=r"D:\SUN3D\SUNRGBD\\"
DepthReader=DepthReader.Reader(MainDir=DataDir,MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)

for i in range(100):
    Imgs, Depth, XYZ, ROI = DepthReader.LoadBatch()
#--------------------------- Create logs files for saving loss during training--------------------------------------------------------------------------------------------------------
    for i in range(Imgs.shape[0]):
             print("DepthRange",Depth.max())
             Depth[i]=(255*Depth[i]/Depth[i].max()).astype(np.uint8)
             vis.show(Imgs[i])
             vis.show(Depth[i],"Depth MaxVal = "+str(Depth[i].max()))
             vis.show(ROI[i]*255, "ROI MaxVal = " + str(ROI[i].max()))
             vis.ShowXYZMap(XYZ[i], "XYZ Map")

             vis.DisplayPointCloud(XYZ[i],Imgs[i],ROI[i],step=2)
    # np.asarray(rgbd_image.color).shape
    #o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault).intrinsic_matrix
    # Flip it, otherwise the pointcloud will be upside down
