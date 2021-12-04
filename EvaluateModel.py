# Evaluate next XYZ and mask predictions
#...............................Imports..................................................................
import os
import ChamferDistance
import numpy
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import torch.nn as nn
import torch.nn.functional as F
import LossFunctions
import SUN3DReader as DepthXYZReader
import Visuallization as vis
import cv2



############################CREATE XYZ loss class ########################################################################################################

LossXYZ=LossFunctions.Loss()

##################################Input paramaters#########################################################################################

#.................................Main Input parametrs...........................................................................................
Sun3EvalDDir=r"/media/breakeroftime/2T/Data_zoo/SUN_RGBD/SUNRGBDLSUNTest///" # Input test folder
#TestFolder = r"Datasets/TranProteus/RealSense/Data//"
#TestFolder = r"TranProteus/Training/FlatLiquidAll/"
Trained_model_path =  r"logs//Defult.torch" # Trained model path
# #Trained_model_path =  r"logsX5//.torch" # Trained model path

MaxSize=1000# max image dimension
UseChamfer=False# Evaluate chamfer distance (this takes lots of time)
#SetNormalizationUsing="ContentXYZ"


#===========================================================================================================================================

# https://arxiv.org/pdf/1406.2283.pdf

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # choose gpu or cpu

MinSize=280 # Min image dimension (height or width)
MaxSize=1000# Max image dimension (height or width)
MaxPixels=800*800*3# Max pixels in a batch (not in image), reduce to solve out if memory problems
MaxBatchSize=4# Max images in batch

#=========================Load net weights====================================================================================================================

#---------------------Create and Initiate net and load net------------------------------------------------------------------------------------
Net=NET_FCN.Net() # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.to(device).eval()

#----------------------------------------Create reader for data sets--------------------------------------------------------------------------------------------------------------
Reader=DepthXYZReader.Reader(MainDir=Sun3EvalDDir,MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=False)


#-------------------------------Create Evaluation statistics dictionary for XYZ--------------------------------------------------------------------
# https://arxiv.org/pdf/1406.2283.pdf
##https://cseweb.ucsd.edu//~haosu/papers/SI2PC_arxiv_submit.pdf
EvalTypes=["RMSE","MAE","TSS",r"MeanError//GTMaxDist","MeanError//stdv","MeanError//MAD","SumPixels"] # TSS total sum of squares, RSS Sum of Squares residuals
if UseChamfer: EvalTypes+=["ChamferDist//GT_MaxDst","ChamferDist//GT_STDV","ChamferDist//GT_Max_Distance"]
StatXYZ={} # Sum All statistics across

for et in EvalTypes:
        StatXYZ[et]=0
# #---------------------Create reader------------------------------------------------------------------------------
#
# GT0 = Reader.LoadSingle()

#-----------------Start evaluation---------------------------------------------------------------------------------
while (Reader.epoch==0): # Test 100 example or one epoch
    Imgs, GTDepth, GT_XYZ, ROI= Reader.LoadSingle() # Load example

    print("------------------------------", Reader.itr, "------------------------------------------------")
    #***************************************************************************************************
#***************************************************************************************************
    # for i in range(Imgs.shape[0]):
    #     print("DepthRange", GTDepth.max())
    #     Depth = (255 * GTDepth[i] / GTDepth[i].max()).astype(np.uint8)
    #     vis.show(Imgs[i])
    #     vis.show(Depth, "Depth MaxVal = " + str(GTDepth[i].max()))
    #     vis.show(ROI[i] * 255, "ROI MaxVal = " + str(ROI[i].max()))
    #     vis.ShowXYZMap(GT_XYZ[i], "XYZ Map")
    #
    #     vis.DisplayPointCloud(GT_XYZ[i], Imgs[i], ROI[i], step=2)
    #*****************************************************************************
 # #  *****************************************************************************
    print("RUN PREDICITION")

    with torch.no_grad():
        PrdXYZ = Net.forward(Images=Imgs, TrainMode=False)  # Run net inference and get prediction

####################################################################################################

       #------------------------ROI Punish depth prediction only within mask----------------------------------------------------

    ROI = torch.autograd.Variable(torch.from_numpy(ROI).unsqueeze(1).cuda(),requires_grad=False)
  # ROI = nn.functional.interpolate(ROI, tuple((PrdXYZ[nm].shape[2], PrdXYZ[nm].shape[3])),mode='bilinear', align_corners=False)
    ROI[ROI < 0.9] = 0
    ROI[ROI > 0.9] = 1
       #-------------------------------Convert GT XYZ map to torch -----------------------------------------------------------------------------------------------------
    GTXYZ_PT = torch.from_numpy(GT_XYZ).cuda().transpose(1,3).transpose(2,3) ###.unsqueeze(1)
 #   PrdXYZ = nn.functional.interpolate(PrdXYZ, tuple(( GTXYZ_PT.shape[2],  GTXYZ_PT.shape[3])), mode='bilinear', align_corners=False)
       #----------------Caclulate relative scale and translation (if not previously calculated----------------------------------------

    with torch.no_grad():
                CatLoss, NormConst= LossXYZ.DiffrentialLoss(PrdXYZ, GTXYZ_PT, ROI) # Calculate relative scqale
                ROI=torch.cat([ROI[0],ROI[0],ROI[0]],0)
                Translation = ((GTXYZ_PT[0]-PrdXYZ[0]/NormConst[0])*ROI).sum(1).sum(1)/ROI.sum(1).sum(1) # Calculate translation between object
                      # else: # in case of predicting 3D model of the content using 3D model of the vessel
                      #     NormConst=[1]
                      #     Translation=np.array([0,0,0])
#######################################################################################################################################################3
    print("Normalization Constant",NormConst)
    PXYZ = PrdXYZ #nn.functional.interpolate(PrdXYZ, tuple((ROI.shape[1],ROI.shape[2])), mode='bilinear', align_corners=False) # predicted XYZ map
    PXYZ = PXYZ[0].cpu().detach().numpy()
    PXYZ=np.moveaxis(PXYZ,[0,1,2],[2,0,1])#swapaxes(PXYZ,[2,0,1])
    GXYZ= GT_XYZ[0] # GT XYZ map
    ROI= ROI[0].data.cpu().numpy()# Mask of the object evaluation will be done only for pixels belonging to this mask
    SumPixels=ROI.sum()

    if ROI.max()>0:


  #-----------------------------------------------------------------------------------------
               #print(NormConst[0])
               # MasksForDisplay.append(ROI)
               # # MasksForDisplay.append(ROI)
               # XYZForDisplay.append(GXYZ)
               # # XYZForDisplay.append(PXYZ)

  #######################Calculatee distances##############################################################################
    # if NormConst[0] < 0.01 or NormConst[0] > 1000:
    #     xx = 0
               for i in range(3):
                   PXYZ[:, :, i] = (PXYZ[:, :, i] / NormConst[0] + Translation.tolist()[i])*ROI # Rescale predicted XYZ map and translate to match GT match (only in the ROI region)
                   GXYZ[:, :, i] *= ROI # Remove predictions outside of ROI region

               #PXYZ=np.fliplr(PXYZ)
               # if np.isnan(gt).any() or np.isnan(prd).any():
               #     xx=0
               dif=np.sum((PXYZ- GXYZ)**2,2)**0.5 # Pygtagorian distance between XYZ points in the sampe pixel
               #dif = np.sum(np.abs(PXYZ - GXYZ), 2)
            #   dif = np.abs(PXYZ[:,:,0] - GXYZ[:,:,0])
               tmpGXYZ=GXYZ.copy()
            #   vis.show(ROI*100,nm+" ROI")
               for i in range(3): tmpGXYZ[:,:,i]= (GXYZ[:,:,i]- (GXYZ[:,:,i].sum()/SumPixels))*ROI # Substract the mean to get deviation from center

               StatXYZ["TSS"] += ((tmpGXYZ)**2).sum()# Total sum of sqr distance from the mean
               mdv = np.abs(tmpGXYZ).sum() / SumPixels # mean absulote deviation
               sdv = np.sqrt((tmpGXYZ**2).sum(2)).sum() / SumPixels # Standart deviation

               # if np.isnan( StatXYZ[nm]["TSS"]).any():
               #     xx = 0

               StatXYZ["SumPixels"] += SumPixels
               StatXYZ["RMSE"]+=(dif**2).sum()
               #StatXYZ[nm]["RMSElog"][i] += (np.log(prd/(GXYZ**2+0.000001))**2).sum()
               SumDif= dif.sum()
               StatXYZ["MAE"]+=SumDif # Mean absoulte error
               dst=0

               #---------------max distance between ppont in the moder
               for i in range(3): #
                   if GXYZ[:, :, i].max() > 0:
                       GXYZ[:, :, i][ROI == 0] = GXYZ[:, :, i].mean()  # Prevent zero from being the minimum since zero mean out of ROI
                   dst += (GXYZ[:,:,i].max()-GXYZ[:,:,i].min())**2
               dst=dst**0.5 # Max distance between points in the model


               StatXYZ["MeanError//GTMaxDist"] += SumDif/  (dst+ 0.00000001)
               StatXYZ["MeanError//stdv"] += SumDif/ (sdv + 0.00000001)
               StatXYZ["MeanError//MAD"] += SumDif /  (mdv + 0.00000001)
               if UseChamfer: # Calculate chamfer distace
                   AbsChamferDist, SqrChamferDist = ChamferDistance.ChamferDistance(GXYZ, PXYZ, ROI)
                   AbsChamferDist*=SumPixels
                   SqrChamferDist*=SumPixels
                   StatXYZ["ChamferDist//GT_MaxDst"] += AbsChamferDist / (dst+0.00001) # normalize chamfer distance by max distance between points in the mdoe
                   StatXYZ["ChamferDist//GT_STDV"] += AbsChamferDist / (sdv+0.00001)# normalize chamfer distance by standart deviation in GT XYZ model
                   StatXYZ["ChamferDist//GT_Max_Distance"] += AbsChamferDist / (mdv + 0.00000001) # normalize chamfer distance by meandeviation in GT XYZ model

#******************************************************************************************************************************************
               # if nm=="ContentXYZ":
               #     MasksForDisplay.append(ROI)
               #     # MasksForDisplay.append(ROI)
               #     XYZForDisplay.append(GXYZ)
               #     # XYZForDisplay.append(PXYZ)
               #     MasksForDisplay.append((dif<(dst)/3)*ROI)
               #     MasksForDisplay.append((dif > (dst) / 3) * ROI)
               #     XYZForDisplay.append(PXYZ)
               #     XYZForDisplay.append(PXYZ)
               #    print(StatXYZ[nm]["AbsRelDif"][i] / StatXYZ[nm]["SumPixels"][i].sum())
   # vis.DisplayPointClouds2(GT["VesselWithContentRGB"][0],XYZForDisplay,MasksForDisplay)
    # https://arxiv.org/pdf/1406.2283.pdf


######################Display final statistics XYZ#################################################################################################

    print("\n\n\n########################   3D XYZ statitics ################################\n\n\n")

    if StatXYZ["SumPixels"] == 0: continue
    # for i,xyz in enumerate(['x','y','z']):
    #     print("\n---------"+xyz+"-----------------\n")
    SumPix=StatXYZ["SumPixels"]

    for et in EvalTypes:
        if et=="SumPixels": continue
        Pr=1
        if "RMSE" in et: Pr=0.5
        print("\n",et,"\t=\t",(StatXYZ[et]/SumPix)**Pr)

    Rsqr=1 - StatXYZ["RMSE"]/StatXYZ["TSS"]
    print("\n","\tR square\t=\t",Rsqr)
    print("\n---------All-----------------\n")
