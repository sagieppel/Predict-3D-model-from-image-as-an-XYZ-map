#  Fully convolutional net that receive image and predict XYZ maps (3 layers per image) and segmentation maps (2 layers).
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Visuallization as vis
######################################################################################################################333
class Net(nn.Module):
########################################################################################################################
    def __init__(self): # MaskList is list of segmentation mask to predict, XYZList is list of XYZ map to predict

        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  encoder----------------------------------------------------------

            self.Encoder = models.resnet101(pretrained=True)

            # ---------------Create Pyramid Scene Parsing PSP layer -------------------------------------------------------------------------
         #    self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]
            # self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
            #
            # for Ps in self.PSPScales:
            #     self.PSPLayers.append(nn.Sequential(
            #         nn.Conv2d(2048, 512, stride=1, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(512),nn.ReLU()))
#---------------------------------Dilated convolution ASPP layers (same as deep lab)------------------------------------------------------------------------------

            self.ASPPScales = [1, 2, 4, 12, 16]
            self.ASPPLayers = nn.ModuleList()
            for scale in self.ASPPScales:
                    self.ASPPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 512, stride=1, kernel_size=3,  padding = (scale, scale), dilation = (scale, scale), bias=False),nn.BatchNorm2d(512),nn.ReLU()))

#)
#-------------------------------------Squeeze ASPP Layer------------------------------------------------------------------------------
            self.SqueezeLayers = nn.Sequential(
                nn.Conv2d(2560, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()#,
                # nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(512),
                # nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))
            # ------------------Skip connection squeeze applied to the (concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 128, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))

            # ----------------Final prediction XYZ maps------------------------------------------------------------------------------------------
            self.XYZLayer=nn.Conv2d(256, 3, stride=1, kernel_size=3, padding=1, bias=False)

##########################################################################################################################################################
    def forward(self, Images,  TrainMode=True, FreezeBatchNorm_EvalON=False):
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
               # ----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]

                #if TrainMode == True:
                tp = torch.FloatTensor # Training mode
                # else:
                #    tp = torch.half
                #    #      self.eval()
                #    self.half()
                if FreezeBatchNorm_EvalON: self.eval() # dont Update batch nor mstatiticls

                # Convert input to pytorch
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(np.float32)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)

# ---------------Convert to cuda gpu-------------------------------------------------------------------------------------------------------------------


                InpImages = InpImages.to(device)
                self.to(device)

#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                EncoderMap = self.Encoder.layer4(x)

#---------------------------------ASPP Layers (Dilated conv)--------------------------------------------------------------------------------
                ASPPFeatures = []  # Results of various of scaled procceessing
                for ASPPLayer in self.ASPPLayers:
                    y = ASPPLayer( EncoderMap )
                    ASPPFeatures.append(y)
                x = torch.cat(ASPPFeatures, dim=1)
                x = self.SqueezeLayers(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear',align_corners=False)  # Upsample
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1) # Apply skip connection and concat with upsample
                  x = self.SqueezeUpsample[i](x) # Squeeze
    # ---------------------------------Final XYZ map prediction-------------------------------------------------------------------------------
                x = self.XYZLayer(x)
                if not TrainMode:# If training predict small version of the the depth map to save memory else predict depth map in image size
                      XYZMap = nn.functional.interpolate(x, size=InpImages.shape[2:4], mode='bilinear',align_corners=False)  # Resize to original image size
                else:
                     XYZMap=x
                return XYZMap
