## Reader for SUN 3D Dataset
import os
import cv2
import threading
import numpy as np
import open3d as o3d
import RGBD_To_XYZMAP
import Visuallization as vis

#########################################################################################################################
class Reader:
    # Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"",  MaxBatchSize=100, MinSize=250, MaxSize=1000, MaxPixels=800 * 800 * 5,TrainingMode=True):
        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and height in pixels
        self.MaxSize = MaxSize  # Max image width and height in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve  out of memory issues)
        self.epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        # ----------------------------------------Create list of annotations arranged by class--------------------------------------------------------------------------------------------------------------
        self.AnnList = []  # Image/annotation list


        print("Creating annotation list for reader this might take a while")

        for sbdir1 in os.listdir(MainDir): # List of all example
            path1 = MainDir +"//" +sbdir1+"//"
            if not os.path.isdir(path1): continue
            for sbdir2 in os.listdir(path1):  # List of all example
                    path2 = path1 + "//" + sbdir2 + "//"
                    if not os.path.isdir(path2): continue
                    for sbdir3 in os.listdir(path2):  # List of all example
                        path3 = path2 + "//" + sbdir3 + "//"
                        if not os.path.isdir(path3): continue
                        if os.path.exists(path3+"//depth//") and os.path.exists(path3+"//image//"):
                                Ann={}
                                for nm in os.listdir(path3+"//depth//"):
                                    if ".png" in nm: Ann["Depth"] = path3 + "//depth//" + nm
                                for nm in os.listdir(path3+"//image//"):
                                    if ".jpg" in nm: Ann["RGB"] = path3 + "//image//" + nm
                                Ann["Scene1"] = sbdir1
                                Ann["Scene2"] = sbdir2
                                self.AnnList.append(Ann)
                        else:
                            for sbdir4 in os.listdir(path3):  # List of all example
                                path4 = path3 + "//" + sbdir4 + "//"
                                if os.path.exists(path4 + "//depth//") and os.path.exists(path4 + "//image//"):
                                        Ann = {}
                                        for nm in os.listdir(path4 + "//depth//"):
                                            if ".png" in nm: Ann["Depth"] = path4 + "//depth//" + nm
                                        for nm in os.listdir(path4 + "//image//"):
                                            if ".jpg" in nm: Ann["RGB"] = path4 + "//image//" + nm
                                        Ann["Scene1"] = sbdir1
                                        Ann["Scene2"] = sbdir2
                                        self.AnnList.append(Ann)
        # ------------------------------------------------------------------------------------------------------------
        print("Done making file list Total=" + str(len(self.AnnList)))
        self.StartLoadBatch()  # Start loading semantic maps batch (multi threaded)

#############################################################################################################################

# Crop and resize image and mask and ROI to feet batch size

#############################################################################################################################
# Crop and resize image and maps and ROI to feet batch size
    def CropResize(self, Maps, Hb, Wb):
            # ========================resize image if it too small to the batch size==================================================================================
            h, w = Maps["ROI"].shape
            Bs = np.min((h / Hb, w / Wb))
            if (Bs < 1):# or Bs > 3 or np.random.rand() < 0.2):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
                h = int(h / Bs) + 1
                w = int(w / Bs) + 1
                for nm in Maps:
                    if hasattr(Maps[nm], "shape"):  # check if array
                        if "Img" in nm:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                        else:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

            # =======================Crop image to fit batch size around center===================================================================================

            if w > Wb:
                X0 =  int((w - Wb)/2-0.1)#np.random.randint(0,w - Wb) #
            else:
                X0 = 0
            if h > Hb:
                Y0 =  int((h - Hb)/2-0.1)#np.random.randint(0,h - Hb) #
            else:
                Y0 = 0

            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    Maps[nm] = Maps[nm][Y0:Y0 + Hb, X0:X0 + Wb]

            # -------------------If still not batch size resize again--------------------------------------------
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                        Maps[nm] = cv2.resize(Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            return Maps

######################################################Augmented Image##################################################################################################################################

    def Augment(self, Img):
            if np.random.rand()<0.01:
              Img = cv2.GaussianBlur( Img, (5, 5), 0)
            if np.random.rand()<0.5:
                Img=Img[:, :, ::-1]
            if np.random.rand() < 0.1:  # Dark light
                        Img *= (0.7 + np.random.rand() * 0.4)
                        Img[ Img > 255 ] = 255

            if np.random.rand() < 0.1:  # GreyScale
                        Gr = Img.mean(axis=2)
                        r = np.random.rand()

                        Img[:, :, 0] =  Img[:, :, 0] * r + Gr * (1 - r)
                        Img[:, :, 1] =  Img[:, :, 1] * r + Gr * (1 - r)
                        Img[:, :, 2] =  Img[:, :, 2] * r + Gr * (1 - r)

            return Img

     ##################################################################################################################################################################

    # Read single image and annotation into batch

    def LoadNext(self, pos, Hb, Wb):
        # -----------------------------------select image-----------------------------------------------------------------------------------------------------
        Ann = self.AnnList[np.random.randint(len(self.AnnList))]
#        print(Ann["RGB"])

        color_raw = o3d.io.read_image(Ann["RGB"])
        depth_raw = o3d.io.read_image(Ann["Depth"])
        Data={}
        Data["Img"] = np.asarray(color_raw).astype(np.float32)  # [:, :, ::-1]

        if (Data["Img"].ndim == 2):  # If grayscale turn to rgb
                Data["Img"] = np.expand_dims(Data["Img"], 3)
                Data["Img"] = np.concatenate([Data["Img"], Data["Img"], Data["Img"]], axis=2)
        if np.random.rand()<0.5:
           Data["Img"] = self.Augment(Data["Img"])
        rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw) # Convert to depth

        Data["XYZ"], Data["ROI"], ImGrey, Data["DepthMap"] = RGBD_To_XYZMAP.RGBD2XYZ(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        Data = self.CropResize(Data, Hb, Wb)


        # ----------------------Generate forward and background segment mask-----------------------------------------------------------------------------------------------------------

        self.BDepth[pos] = Data["DepthMap"]
        self.BRGB[pos] = Data["Img"]
        self.BXYZ[pos] = Data["XYZ"]
        self.BROI[pos] = Data["ROI"]

############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            self.Width = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight #900
            self.Height = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width #900
            if self.Height * self.Width < self.MaxPixels: break
        self.BatchSize = np.int(np.min([(np.floor(self.MaxPixels / self.Height * self.Width)), self.MaxBatchSize]))
        # ===================Create empty batch ===========================================================
        self.BRGB = np.zeros([self.BatchSize, self.Height, self.Width, 3], dtype=np.float32)
        self.BDepth = np.zeros([self.BatchSize, self.Height, self.Width], dtype=np.float32)
        self.BXYZ = np.zeros([self.BatchSize, self.Height, self.Width, 3], dtype=np.float32)
        self.BROI = np.zeros([self.BatchSize, self.Height, self.Width], dtype=np.float32)
        # ====================Start reading data multithreaded===================================================
        self.thread_list = []
        for pos in range(self.BatchSize):
            th = threading.Thread(target=self.LoadNext, name="threadReader" + str(pos), args=(pos, self.Height, self.Width))
            self.thread_list.append(th)
            th.start()

    ###########################################################################################################
    # Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
        for th in self.thread_list:
            th.join()

    ########################################################################################################################################################################################
    def LoadBatch(self):
        # Load batch for training (muti threaded  run in parallel with the training proccess)
        # return previously  loaded batch and start loading new batch
        self.WaitLoadBatch()
        Imgs = self.BRGB
        Depth = self.BDepth
        XYZ = self.BXYZ
        ROI = self.BROI
        self.StartLoadBatch()
        return  Imgs, Depth, XYZ, ROI

    ########################################################################################################################################################################################
    def LoadSingle(self,Hb=-1,Wb=-1):
        # -----------------------------------select image-----------------------------------------------------------------------------------------------------
        if self.itr>=len(self.AnnList):
            self.itr=0
            self.epoch+=1
        self.itr+=1
        Ann = self.AnnList[self.itr]
        #        print(Ann["RGB"])

        color_raw = o3d.io.read_image(Ann["RGB"])
        depth_raw = o3d.io.read_image(Ann["Depth"])
        Data = {}
        Data["Img"] = np.asarray(color_raw).astype(np.float32)  # [:, :, ::-1]

        if (Data["Img"].ndim == 2):  # If grayscale turn to rgb
            Data["Img"] = np.expand_dims(Data["Img"], 3)
            Data["Img"] = np.concatenate([Data["Img"], Data["Img"], Data["Img"]], axis=2)
        rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)

        Data["XYZ"], Data["ROI"], ImGrey, Data["DepthMap"] = RGBD_To_XYZMAP.RGBD2XYZ(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        if Hb>0 and Wb>0:
               Data = self.CropResize(Data, Hb, Wb)
        # vis.ShowXYZMap(XYZ, "XYZ Map")
        # vis.DisplayPointCloud(XYZ, Img, ROI, step=2)

        # -----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
        #  self.before = Maps['VesselWithContentRGB'].copy()
        # Maps = self.Augment(Msk)
        # if Hb != -1:
        #     Maps = self.CropResize(Maps, Hb, Wb)

        #   self.after=np.hstack([cv2.resize(self.before,(Wb,Hb)),Maps['VesselWithContentRGB'].copy()])
        for nm in Data: # Create batch structure
            Data[nm]=np.expand_dims(Data[nm],axis=0)

        return Data["Img"],Data["DepthMap"],Data["XYZ"],Data["ROI"]
    # ########################################################################################################################################################################################
    # def LoadSingleResized(self,Hb=-1,Wb=-1):
    #     # -----------------------------------select image-----------------------------------------------------------------------------------------------------
    #     if self.itr>=len(self.AnnList):
    #         self.itr=0
    #         self.epoch+=1
    #     self.itr+=1
    #     Ann = self.AnnList[self.itr]
    #     #        print(Ann["RGB"])
    #
    #     color_raw = o3d.io.read_image(Ann["RGB"])
    #     depth_raw = o3d.io.read_image(Ann["Depth"])
    #     Data = {}
    #     Data["Img"] = np.asarray(color_raw).astype(np.float32)  # [:, :, ::-1]
    #
    #     if (Data["Img"].ndim == 2):  # If grayscale turn to rgb
    #         Data["Img"] = np.expand_dims(Data["Img"], 3)
    #         Data["Img"] = np.concatenate([Data["Img"], Data["Img"], Data["Img"]], axis=2)
    #     rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)
    #
    #     # print(rgbd_image)
    #     #
    #     # plt.subplot(1, 2, 1)
    #     # plt.title('Redwood grayscale image')
    #     # plt.imshow(rgbd_image.color)
    #     # plt.subplot(1, 2, 2)
    #     # plt.title('Redwood depth image')
    #     # plt.imshow(rgbd_image.depth)
    #     # plt.show()
    #     # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    #     # Flip it, otherwise the pointcloud will be upside down
    #     # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #     # o3d.visualization.draw_geometries([pcd])
    #
    #     Data["XYZ"], Data["ROI"], ImGrey, Data["DepthMap"] = RGBD_To_XYZMAP.RGBD2XYZ(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    #     #if Hb>0 and Wb>0:
    #    # Data = self.CropResize(Data, Hb=int(Data["Img"].shape[0]*0.5), Wb=int(Data["Img"].shape[1]*0.5))
    #     hh = int(Data["Img"].shape[0] * 0.5)
    #     ww = int(Data["Img"].shape[1] * 0.5)
    #     for nm in Data:
    #         Data[nm] = Data[nm][:hh, :ww]
    #     Data = self.CropResize(Data, Hb=int(Data["Img"].shape[0] * 2), Wb=int(Data["Img"].shape[1] * 2))
    #     # vis.ShowXYZMap(XYZ, "XYZ Map")
    #     # vis.DisplayPointCloud(XYZ, Img, ROI, step=2)
    #
    #     # -----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
    #     #  self.before = Maps['VesselWithContentRGB'].copy()
    #     # Maps = self.Augment(Msk)
    #     # if Hb != -1:
    #     #     Maps = self.CropResize(Maps, Hb, Wb)
    #
    #     #   self.after=np.hstack([cv2.resize(self.before,(Wb,Hb)),Maps['VesselWithContentRGB'].copy()])
    #     for nm in Data: # Create batch structure
    #         Data[nm]=np.expand_dims(Data[nm],axis=0)
    #
    #     return Data["Img"],Data["DepthMap"],Data["XYZ"],Data["ROI"]
