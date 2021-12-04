import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.image as mpimg
##########################################################################################
def ResizeToMaxSize(Im,MaxSize):
        h=Im.shape[0]
        w=Im.shape[1]

        r=np.min([MaxSize/h,MaxSize/w])
        if r<1:
           Im=cv2.resize(Im,(int(r*w),int(r*h)))
        return Im
##########################################################################################
def ResizeToScreen(Im):
        h=Im.shape[0]
        w=Im.shape[1]
        r=np.min([1000/h,1800/w])
        Im=cv2.resize(Im,(int(r*w),int(r*h)))
        return Im
########################################################################################
def showcv2(Im,txt=""):
    cv2.destroyAllWindows()
   # print("IM text")
   # print(txt)
    cv2.imshow(txt,ResizeToScreen(Im.astype(np.uint8)))
  #  cv2.moveWindow(txt, 1, 1);
    ch=cv2.waitKey()
    cv2.destroyAllWindows()
   # cv2.destroyAllWindows()
    return ch
########################################################################################
def show(Im,txt=""):
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if np.ndim(Im)==3:
        plt.imshow(Im[:, :, ::-1].astype(np.uint8))
    else:
        plt.imshow(Im.astype(np.uint8))
    plt.title(txt)
    plt.show()
########################################################################################
def trshow(Im,txt=""):
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.imshow((Im.data.cpu().numpy()).astype(np.uint8))
    plt.title(txt)
    plt.show()
#############################################################################3
def GreyScaleToRGB(Img):
    I=np.expand_dims(Img,2)
    rgb=np.concatenate([I,I,I],axis=2)
    return  rgb
################################################################################################
def DisplayPointCloud(xyzMap,RGB,ROI,step=5):
    # -----------------------------3d point cloud----------------------------------------------------------
    NumPoints=int(xyzMap.shape[0]*xyzMap.shape[1]/step**2)
    xyz = np.zeros([NumPoints, 3], np.float32)
    colors = np.zeros([NumPoints, 3], np.float32)
    tt=0
    for x in range(0,xyzMap.shape[1],step):
        for y in range(0,  xyzMap.shape[0],step):
            if (ROI[y, x]) > 0 and  tt<NumPoints:
                xyz[tt]  = xyzMap[y, x]
                colors[tt] = RGB[y, x]/255
                tt+=1
               # print(tt)
    xyz = xyz[:tt,:]
    colors = colors[:tt, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


################################################################################################
def DisplayPointClouds2(img,xyzMap,Masks):
    # -----------------------------3d point cloud----------------------------------------------------------
    XYZ2Color = [[255, 0, 0],[0, 255, 0], [0, 0, 255],[255, 255, 0],[0, 255, 255], [255, 0, 255], [255, 125, 65],[80, 50, 80]]
    xyz = np.zeros([20000, 3], np.float32)
    colors = np.zeros([20000, 3], np.float32)
    # "VesselXYZ"  "ContentXYZ"  "VesselOpening_XYZ"


    show(img.astype(np.uint8))

    tt=0
    while True:
        print("collecting points", tt)
        nm =np.random.randint(len(xyzMap))
        # nm = "VesselOpening_XYZ"
        x = np.random.randint(xyzMap[nm].shape[1])
        y = np.random.randint(xyzMap[nm].shape[0])
        print("dddddd")
        if (Masks[nm][y, x]) > 0.95:  # *GT["ROI"][0,y,x]
            print("EEEEEEEe")
            if np.abs(xyzMap[nm][y, x]).sum() > 0:  # and (GT[nm.replace("XYZ","Mask")][0,y,x]*GT["ROI"][0,y,x])>0.6:
                print("ffffffffff")
                xyz[tt] = xyzMap[nm][y, x]
                colors[tt] = XYZ2Color[nm]
                tt += 1
                if tt >= xyz.shape[0]-10000: break

    while True:
        x = np.random.randint(xyzMap[nm].shape[1])
        y = np.random.randint(xyzMap[nm].shape[0])
        if (Masks[2][y, x]) > 0.95:  # *GT["ROI"][0,y,x]
            while(True):
                y = np.random.randint(xyzMap[nm].shape[1])
                if np.abs(xyzMap[nm][y, x]).sum() > 0:  # and (GT[nm.replace("XYZ","Mask")][0,y,x]*GT["ROI"][0,y,x])>0.6:
                    xyz[tt] = xyzMap[2][y, x]
                    colors[tt] = [0,0,0]
                    tt += 1
                    xyz[tt] = xyzMap[0][y, x]
                    colors[tt] = [125, 0, 125]
                    tt += 1
                    if tt >= xyz.shape[0]: break
        if tt >= xyz.shape[0]: break


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
########################################################################################
def ShowXYZMap(XYZMap,txt=""):
    XYZ=np.zeros_like(XYZMap)
    for i in range(3):
            XYZ[:,:,i]=XYZMap[:,:,i]-XYZMap[:,:,i].min()
    XYZ=(XYZ/(XYZ.max()-XYZ.min())*255).astype(np.uint8)
    show(XYZ,txt)
