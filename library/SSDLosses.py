###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
from library.ssd import jaccard, intersect 
import numpy as np

class SSDSingleClassLoss(nn.Module):
    """SSD Loss Function
    Compute Targets:
        1) Produce indices for positive matches by matching ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
           
        2) Calculates location and confidence loss for positive matches 

        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
               - Negative match background CFs are sorted in ascending order (less confident pred. first)
               - If Positive match exists
                   - Nneg is calculated by Mining_Neg2PosRatio * Npos, clipped below with min_NegMiningSample
                   - Smallest Nneg background CFs are selected, CF's above maxBackroundCFforLossCalc are ommitted and used in loss calc
               - If there is no positive match, min_NegMiningSample less confident background CFs are taken in to loss
           
    Objective Loss:
        L(x,c,l,g) = [(LconfPosMatch(x, c)) / Npos] +
                     [(λ * LconfNegMatch(x, c)) / Nneg] + [(α*Lloc(x,l,g)) / Npos]
        
        
        Where, LconfPosMatch is the log softmax person class conf loss of positive matched boxes,
        LconfNegMatch is the log softmax background class conf loss of negative matched boxes,
        Lloc is the SmoothL1 Loss weighted by α which is set to 1 by cross val for original multiclass SSD.
        
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            Npos: number of matched default boxes
            Neg: number of negative matches used in loss function after negative mining
            x: positive match selector
    """

    def __init__(self, Anchor_box_wh, Anchor_box_xy, alpha = 1, Jaccardtreshold = 0.5, 
                 Mining_Neg2PosRatio = 6, min_NegMiningSample = 10, maxBackroundCFforLossCalc = 0.5, negConfLosslambda = 1.0,
                regularizedLayers = None):
        '''
        Args:
        Anchor_box_wh: (tensor) Anchor boxes (cx,cy, w, h) form in original image, Shape: [numPreds=5376,4]
        Anchor_box_xy: (tensor) Anchor boxes (cxmin,cymin, xmax, ymax) form in original image, Shape: [numPreds=5376,4]
        '''
        
        super(SSDSingleClassLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.Anchor_box_wh = Anchor_box_wh
        self.Anchor_box_xy = Anchor_box_xy
        self.alpha = alpha
        self.Jaccardtreshold = Jaccardtreshold
        
        self.Mining_Neg2PosRatio = Mining_Neg2PosRatio
        self.min_NegMiningSample = min_NegMiningSample
        self.maxBackroundCFforLossCalc = maxBackroundCFforLossCalc
        self.negConfLosslambda = negConfLosslambda
        
        self.regularizedLayers = regularizedLayers
        
        # application specific variances for SSD
        self.var_x = 0.1
        self.var_y = 0.1
        self.var_w = 0.2
        self.var_h = 0.2
        

    def forward(self, pred_box_delt, pred_CF ,GT_box_wh, model= None):
        """Multibox Loss
        Args:
            pred_box_delt : (tensor) Location predictions in delta form (dcx, dcy, dw, dh), shape[numPreds=5376,4] 
            pred_CF : (tensor) Confidence predictions (person, nonperson), shape[numPreds=5376,2] 
            GT_box_wh : (tensor) Ground truth boxes in (xmin, ymin, w, h) form, shape [numObjects, 4]
        """
        
        
        device =self.device
        alpha = self.alpha 
        Jaccardtreshold = self.Jaccardtreshold
        Mining_Neg2PosRatio = self.Mining_Neg2PosRatio 
        min_NegMiningSample = self.min_NegMiningSample
        maxBackroundCFforLossCalc = self.maxBackroundCFforLossCalc
        negConfLosslambda = self.negConfLosslambda 
        
        reg = torch.tensor(.0).to(device)
        
        if  (len(GT_box_wh)==0): # if there is no labeled person in original image, set location loss to 0
            loss_l = torch.tensor([.0])
            num_pos = 0
        else:
            GT_box_wh = GT_box_wh[:,1:] # first element of GT_box is label of picture, it is deleted
            
            # GT_box_cxcy_wh: GT boxes in (cx, cy, w, h) form, used in ghat calculation
            GT_box_cxcy_wh = GT_box_wh.clone().to(device)
            GT_box_cxcy_wh[:,0] = GT_box_wh[:,0]+GT_box_wh[:,2]/2
            GT_box_cxcy_wh[:,1] = GT_box_wh[:,1]+GT_box_wh[:,3]/2
            
            # GT_box_xy: GT boxes in (xmin, ymin, xmax, ymax) form, used in Jaccard for positive match check
            GT_box_xy = GT_box_wh.detach().clone().to(device)
            GT_box_xy[:,2] = GT_box_wh[:,2] + GT_box_wh[:,0]
            GT_box_xy[:,3] = GT_box_wh[:,3] + GT_box_wh[:,1]

            # Calculate Loss
            JaccardIndices = jaccard(self.Anchor_box_xy,GT_box_xy)
            posMatches = torch.nonzero(JaccardIndices >= Jaccardtreshold)
            negMatchAnchIdx = torch.nonzero(JaccardIndices.max(dim=1).values < Jaccardtreshold).flatten()
            
            # posMatches: tensor[numpreds=5376,2], shows the matches anchor boxes to GT boxes, 
            # first column: ID of matched anchor, second column: ID of GT box
            posMatchAnchIdx = posMatches[:,0]
            posMatchGTIdx = posMatches[:,1]

            pred_backGrCF = pred_CF[:,1]
            negMatch_pred_backGrCF = pred_backGrCF[negMatchAnchIdx]

            
            posMatchAnchs = self.Anchor_box_wh[posMatchAnchIdx]
            num_pos = posMatches.shape[0]

        if num_pos:
            posMatch_pred_box_delt = pred_box_delt[posMatchAnchIdx]
            posMatch_pred_CF = pred_CF[posMatchAnchIdx][:,0]
#             print(f'posMatch_pred_CF: {posMatch_pred_CF}')
            posMatchGTs = GT_box_cxcy_wh[posMatchGTIdx]


            # Calculate g_hat 
            ghat_cx = (posMatchGTs[:,0]-posMatchAnchs[:,0])/posMatchAnchs[:,2]/self.var_x
            ghat_cy = (posMatchGTs[:,1]-posMatchAnchs[:,1])/posMatchAnchs[:,3]/self.var_y
            ghat_w = torch.log(posMatchGTs[:,2]/posMatchAnchs[:,2])/self.var_w
            ghat_h = torch.log(posMatchGTs[:,3]/posMatchAnchs[:,3])/self.var_h
            ghat = torch.cat((ghat_cx.unsqueeze(1), ghat_cy.unsqueeze(1), ghat_w.unsqueeze(1), ghat_h.unsqueeze(1)),dim=1)

            # Calculate location loss
            smoothL1 = torch.nn.SmoothL1Loss(reduction='sum', beta=1.0).to(device)
            ghat_1D = ghat.view(1,-1)
            posMatch_pred_box_delt_1D = posMatch_pred_box_delt.view(1,-1)
            loc_loss = smoothL1(posMatch_pred_box_delt_1D, ghat_1D)

            # Calculate conf loss for positive matches
            posMatch_CF_loss = -torch.log(posMatch_pred_CF).sum()
#             print(f'posMatch_CF_loss: {posMatch_CF_loss}')

            # Hard negative mining
            negMatch_pred_backGrCF,_=negMatch_pred_backGrCF.sort(0, descending=False)
            
            # set hard negative mining sample num  
            # clamp number of negtive samples with min_NegMiningSample below, Neg2Pos Ratio x numPositive number above
            num_hardmined_negative = int(np.max([num_pos*Mining_Neg2PosRatio,min_NegMiningSample]))
            num_hardmined_negative = int(np.min([num_hardmined_negative, negMatch_pred_backGrCF.shape[0]]))
            negMatch_pred_backGrCF_mined = negMatch_pred_backGrCF[0:num_hardmined_negative]
            # select low confidence backround CFs
            negMatch_pred_backGrCF_mined = negMatch_pred_backGrCF_mined[negMatch_pred_backGrCF_mined<maxBackroundCFforLossCalc]
            num_hardmined_negative = negMatch_pred_backGrCF_mined.shape[0]
            
#             print(f'negMatch_pred_backGrCF_mined: {negMatch_pred_backGrCF_mined}')
            negMatch_CF_losses_mined = -torch.log(negMatch_pred_backGrCF_mined) 
            negMatch_CF_loss = negMatch_CF_losses_mined.sum()
            if (num_hardmined_negative == 0):
                negMatch_CF_loss = torch.tensor(.0)
            else:
                negMatch_CF_loss = (negMatch_CF_loss / num_hardmined_negative)*negConfLosslambda
#                 print(f'num_hardmined_negative: {num_hardmined_negative}')
                
#             print(f'negMatch_CF_loss : {negMatch_CF_loss.item()}')
            
            loss_l = alpha*loc_loss / num_pos
        
            posMatch_CF_loss = posMatch_CF_loss / num_pos
            loss_c = (posMatch_CF_loss) + (negMatch_CF_loss)
            
        else:
            # If there is no pos match or there is no labeled person in original image, set loc los to zero
            # calculate confidence loss for minimum number of backgorund classifications 
            
            loss_l = torch.tensor(.0)
            posMatch_CF_loss = torch.tensor(.0)
        
            negCFs_sorted, _ = pred_CF[:,1].view(-1,1).sort(0,descending=False)
            num_hardmined_negative = int(min_NegMiningSample)
            negMatch_pred_backGrCF_mined = negCFs_sorted[0:num_hardmined_negative]
            negMatch_CF_losses_mined = -torch.log(negMatch_pred_backGrCF_mined) 
            negMatch_CF_loss = negMatch_CF_losses_mined.sum()
            negMatch_CF_loss = (negMatch_CF_loss / num_hardmined_negative)*negConfLosslambda
            loss_c = negMatch_CF_loss
            
            # L2 Regularization of specified layers
            if model != None:
                if (self.regularizedLayers != None):
                    for layer,lamb in self.regularizedLayers:
                        layer_attribute = getattr(model, layer)
                        m = layer_attribute.op.weight.numel() + layer_attribute.op.bias.numel()
                        reg += ((layer_attribute.op.bias.view(1,-1)**2).sum() + (layer_attribute.op.weight.view(1,-1)**2).sum())*lamb/m

#             print(f'No Positive Match - Neg Loss is: {loss_c}')
        
#         print(f'loss_l:           {loss_l}')
#         print(f'posMatch_CF_loss: {posMatch_CF_loss}')
#         print(f'negMatch_CF_loss: {negMatch_CF_loss}')
#         print('')
        
        return loss_l + reg, loss_c