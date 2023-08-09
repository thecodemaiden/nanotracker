###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, torchvision, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime
from collections import Counter
import torchvision.ops as ops
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm


import qat_core

def ssd_postprocess_person_cls(pred):
    """
    Take the models prediction outputs (pred) and make the post processing operations to
    classification head outputs since the output is not directly class probabilities.
    Assuming square input image so H=W.
    Assuming binary classes (0/1).

    Input:
    - pred: predicted outputs of the model as list of length 4 as [output1_reg, output1_class, output2_reg, output2_class] 
            and shape of [(NR1, CR1, HR1, WR1), (NC1, CC1, HC1, WC1), (NR2, CR2, HR2, WR2), (NC2, CC2, HC2, WC2)].

    Returns:
    - person_cls: person class probabilities (torch.FloatTensor) shape of [CC1/2*HC1*WC1 + CC2/2*HC2*WC2].
    """
    head_regression_hires     = pred[0]
    head_classification_hires = pred[1]
    head_regression_lores     = pred[2]
    head_classification_lores = pred[3]

    # split classification head outputs for person and background
    head_classification_hires_background = head_classification_hires[0,1::2,:,:]
    head_classification_hires_person     = head_classification_hires[0,0::2,:,:]
    head_classification_lores_background = head_classification_lores[0,1::2,:,:]
    head_classification_lores_person     = head_classification_lores[0,0::2,:,:]

    ## assuming square input image so rows=cols
    ## I'll just define these globally:
    hires_rowscols   = head_regression_hires.shape[3] # could have been classification head too, just getting dimension
    lores_rowscols   = head_regression_lores.shape[3] # could have been classification head too, just getting dimension
    hires_numanchors = int(head_regression_hires.shape[1]/4) # 4 because xywh
    lores_numanchors = int(head_regression_lores.shape[1]/4) # 4 because xywh

    background_hires_flat = explicit_flatten(head_classification_hires_background, 'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    background_lores_flat = explicit_flatten(head_classification_lores_background, 'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    person_hires_flat     = explicit_flatten(head_classification_hires_person,     'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    person_lores_flat     = explicit_flatten(head_classification_lores_person,     'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    
    person_flat     = torch.cat((person_hires_flat, person_lores_flat))
    background_flat = torch.cat((background_hires_flat, background_lores_flat))

    total_cat      = torch.cat( ( torch.unsqueeze(person_flat,0) , torch.unsqueeze(background_flat,0) ) )
    softmax_fcn    = torch.nn.Softmax(dim=0)
    softmax_result = softmax_fcn(total_cat)

    person_hires_flat_sft = softmax_result[0,:][0:background_hires_flat.shape[0]]
    person_lores_flat_sft = softmax_result[0,:][background_hires_flat.shape[0]:]

    person_hires_classification_scores = explicit_unflatten(person_hires_flat_sft, 'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    person_lores_classification_scores = explicit_unflatten(person_lores_flat_sft, 'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)

    person_cls = torch.cat(( person_hires_flat_sft, person_lores_flat_sft ))

    return person_cls


def ssd_postprocess_person_bboxes(pred, image_width, image_height, anchors_head1, anchors_head2):
    """
    Take the models prediction output (pred) and make the post processing operations to
    show bboxes.
    Assuming square input image so H=W.
    Assuming binary classes (0/1).

    Input:
    - pred: predicted outputs of the model as list of length 4[output1_reg, output1_class, output2_reg, output2_class] 
            shape of [(NR1, CR1, HR1, WR1), (NC1, CC1, HC1, WC1), (NR2, CR2, HR2, WR2), (NC2, CC2, HC2, WC2)].
    - image_width:  Integer. 
    - image_height: Integer.
    - anchors_head1: list of length 4, contains image_width/image_height*anchor_ratios as tuples. 
                     shape [(W*A1, H*B1), (W*A2, H*B2), (W*A3, H*B3), (W*A4, H*B4)] where A#num and B#num are
                     corresponding different aspect ratios.
    - anchors_head2: list of length 4, contains image_width/image_height*anchor_ratios as tuples. 
                     shape [(W*C1, H*D1), (W*C2, H*D2), (W*C3, H*D3), (W*C4, H*D4)] where C#num and D#num are
                     corresponding different aspect ratios.
    Returns:
    - absolute_boxes: absolute value of bounding boxes (torch.FloatTensor) shape of [CR1/4*HR1*WR1 + CR2/4*HR2*WR2, 4]. 
    """
    head_regression_hires     = pred[0]
    head_classification_hires = pred[1]
    head_regression_lores     = pred[2]
    head_classification_lores = pred[3]

    ## assuming square input image so rows=cols
    ## I'll just define these globally:
    hires_rowscols   = head_regression_hires.shape[3] # could have been classification head too, just getting dimension
    lores_rowscols   = head_regression_lores.shape[3] # could have been classification head too, just getting dimension
    hires_numanchors = int(head_regression_hires.shape[1]/4) # 4 because xywh
    lores_numanchors = int(head_regression_lores.shape[1]/4) # 4 because xywh

    # Postprocess regression + classification together, i.e., apply NMS
    delta_x_hires = head_regression_hires[0, 0::4, :, :] # skip 4 means skip y,w,h and land on x again
    delta_y_hires = head_regression_hires[0, 1::4, :, :] # skip 4 means skip w,h,x and land on y again, etc...
    delta_w_hires = head_regression_hires[0, 2::4, :, :] 
    delta_h_hires = head_regression_hires[0, 3::4, :, :] 

    delta_x_lores = head_regression_lores[0, 0::4, :, :] # skip 4 means skip y,w,h and land on x again
    delta_y_lores = head_regression_lores[0, 1::4, :, :] # skip 4 means skip w,h,x and land on y again, etc...
    delta_w_lores = head_regression_lores[0, 2::4, :, :] 
    delta_h_lores = head_regression_lores[0, 3::4, :, :]

    
    ## There is also a concept called priorbox variance, see:
    ## https://github.com/weiliu89/caffe/issues/155#issuecomment-243541464
    ## https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    ##
    ## values taken from the xml, see layer "PriorBoxClustered":
    var_x = 0.1
    var_y = 0.1
    var_w = 0.2
    var_h = 0.2

    w_anchors_hires = torch.tensor(anchors_head1)[:,0]
    h_anchors_hires = torch.tensor(anchors_head1)[:,1]
    w_anchors_lores = torch.tensor(anchors_head2)[:,0]
    h_anchors_lores = torch.tensor(anchors_head2)[:,1]

    x_anchors_hires = populate_xy_anchors(delta_x_hires, 'x')
    y_anchors_hires = populate_xy_anchors(delta_y_hires, 'y')
    x_anchors_lores = populate_xy_anchors(delta_x_lores, 'x')
    y_anchors_lores = populate_xy_anchors(delta_y_lores, 'y')

    w_anchors_hires_rpt = populate_wh_anchors(delta_w_hires, w_anchors_hires)
    h_anchors_hires_rpt = populate_wh_anchors(delta_h_hires, h_anchors_hires)
    w_anchors_lores_rpt = populate_wh_anchors(delta_w_lores, w_anchors_lores)
    h_anchors_lores_rpt = populate_wh_anchors(delta_h_lores, h_anchors_lores)

    absolute_x_hires = delta_x_hires * w_anchors_hires_rpt * var_x + x_anchors_hires 
    absolute_y_hires = delta_y_hires * h_anchors_hires_rpt * var_y + y_anchors_hires
    absolute_x_lores = delta_x_lores * w_anchors_lores_rpt * var_x + x_anchors_lores
    absolute_y_lores = delta_y_lores * h_anchors_lores_rpt * var_y + y_anchors_lores

    absolute_w_hires = (delta_w_hires * var_w).exp() * w_anchors_hires_rpt 
    absolute_h_hires = (delta_h_hires * var_h).exp() * h_anchors_hires_rpt
    absolute_w_lores = (delta_w_lores * var_w).exp() * w_anchors_lores_rpt
    absolute_h_lores = (delta_h_lores * var_h).exp() * h_anchors_lores_rpt

    absolute_hires_xleft   = absolute_x_hires - absolute_w_hires/2
    absolute_hires_xright  = absolute_x_hires + absolute_w_hires/2
    absolute_hires_ytop    = absolute_y_hires - absolute_h_hires/2
    absolute_hires_ybottom = absolute_y_hires + absolute_h_hires/2

    absolute_lores_xleft   = absolute_x_lores - absolute_w_lores/2
    absolute_lores_xright  = absolute_x_lores + absolute_w_lores/2
    absolute_lores_ytop    = absolute_y_lores - absolute_h_lores/2
    absolute_lores_ybottom = absolute_y_lores + absolute_h_lores/2

    absolute_hires_xleft_flat   = explicit_flatten(absolute_hires_xleft,   'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_hires_xright_flat  = explicit_flatten(absolute_hires_xright,  'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_hires_ytop_flat    = explicit_flatten(absolute_hires_ytop,    'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_hires_ybottom_flat = explicit_flatten(absolute_hires_ybottom, 'hires', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)

    absolute_lores_xleft_flat   = explicit_flatten(absolute_lores_xleft,   'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_lores_xright_flat  = explicit_flatten(absolute_lores_xright,  'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_lores_ytop_flat    = explicit_flatten(absolute_lores_ytop,    'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)
    absolute_lores_ybottom_flat = explicit_flatten(absolute_lores_ybottom, 'lores', hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors)

    absolute_xleft   = torch.unsqueeze(torch.cat((absolute_hires_xleft_flat,   absolute_lores_xleft_flat))  ,1)
    absolute_xright  = torch.unsqueeze(torch.cat((absolute_hires_xright_flat,  absolute_lores_xright_flat)) ,1)
    absolute_ytop    = torch.unsqueeze(torch.cat((absolute_hires_ytop_flat,    absolute_lores_ytop_flat))   ,1)
    absolute_ybottom = torch.unsqueeze(torch.cat((absolute_hires_ybottom_flat, absolute_lores_ybottom_flat)),1)

    absolute_boxes = torch.cat((absolute_xleft, absolute_ytop, absolute_xright, absolute_ybottom), dim=1)
    return absolute_boxes



# so that we know what goes where
def explicit_flatten(tensor, hires_or_lores, hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors):
    flattened_tensor = torch.zeros_like(tensor.flatten())
    
    if(hires_or_lores=='hires'):
        rc = hires_rowscols
        na = hires_numanchors
    elif(hires_or_lores=='lores'):
        rc = lores_rowscols
        na = lores_numanchors
    else:
        print("somethings wrong")
        return
        
    for row in range(0, rc):
        for col in range(0, rc):
            for anc in range(0, na):
                flattened_tensor[anc*rc*rc + row*rc + col] = tensor[anc,row,col];
                
    return flattened_tensor

# so that we know what goes where
def explicit_unflatten(flattened_tensor, hires_or_lores, hires_rowscols, hires_numanchors, lores_rowscols, lores_numanchors):
    if(hires_or_lores=='hires'):
        tensor = torch.zeros((hires_numanchors, hires_rowscols, hires_rowscols))
        rc = hires_rowscols
        na = hires_numanchors
    elif(hires_or_lores=='lores'):
        tensor = torch.zeros((lores_numanchors, lores_rowscols, lores_rowscols))
        rc = lores_rowscols
        na = lores_numanchors
    else:
        print("somethings wrong")
        return
        
    for row in range(0, rc):
        for col in range(0, rc):
            for anc in range(0, na):
                tensor[anc,row,col] = flattened_tensor[anc*rc*rc + row*rc + col];
                
    return tensor


def plot_softmax_confidence_scores(person_hires_flat_sft, person_lores_flat_sft):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(person_hires_flat_sft.detach().cpu().numpy())
    ax.plot(person_lores_flat_sft.detach().cpu().numpy())
    ax.grid()
    ax.legend(['hires confidences', 'lores confidences'])
    plt.title('softmax-processed confidence scores for the two heads')
    plt.show()


def populate_wh_anchors(delta_ref, wh_anchors_hilores):
    wh_anchors_hilores_rpt = torch.ones_like(delta_ref)
    for i in range(0, wh_anchors_hilores_rpt.shape[0]):
        wh_anchors_hilores_rpt[i] = wh_anchors_hilores_rpt[i]*wh_anchors_hilores[i]

    return wh_anchors_hilores_rpt

def populate_xy_anchors(delta_ref, x_or_y):
    xy_anchors_hilores = torch.zeros_like(delta_ref)
    scale = 512 / delta_ref.shape[2]
    for i in range(0, xy_anchors_hilores.shape[0]): # count anchors
        for j in range(0, xy_anchors_hilores.shape[1]): # count width
            for k in range(0, xy_anchors_hilores.shape[2]): # count height
                if(x_or_y == 'x'):
                    xy_anchors_hilores[i,j,k] = scale * k + (scale +1) / 2 # More precise conversion
                if(x_or_y == 'y'):
                    xy_anchors_hilores[i,j,k] = scale * j + (scale +1) / 2
    return xy_anchors_hilores

def plot_image_mnv2_2xSSDlite(image, pred_person_cls = None, pred_absolute_boxes = None, color = 'r'
                              ,nmsIoUTreshold = 0.45, predConfPlotTreshold = 0.6,target=None, figsize=(16,16),
                              saveFig=False, imageID=None, folderName='UnconstFPT'):
    """ Plots original image, ground truths, and predictions if available. 
    Does non-maximum-suppression and plots perdiction boxes, saves figure under "Training Outputs" folder in specified folderName
    Args:
        image : (Tensor) Shape[Channel,width, height]
        pred_person_cls : (Tensor) person class confidences for predicted boxes Shape[numPred,1]
        pred_absolute_boxes : (Tensor) predicted boxes [xmin,ymin,xmax,ymax] Shape[numPred,4]
        color: Color of drawn predicted boxes
        nmsIoUTreshold : non max suppression IoU treshold
        predConfPlotTreshold : Confidence treshold to draw predicted boxes
        target : (Tensor) Ground truth boxes [pictureID, xmin, ymin, w, h] Shape[numGt, 5]
        folderName : Foldername under ./Model Outputs diectory to save figure. 
        
    Return: none
        
    """
    # if image is normalized to [-1,1], re-map it to [0,1] for plotting purposes
    if (image.min()<0):
        image[0,:,:] = (image[0,:,:]/2)+0.5
        image[1,:,:] = (image[1,:,:]/2)+0.5
        image[2,:,:] = (image[2,:,:]/2)+0.5

    image = image.permute(1, 2, 0).to("cpu")
    fig, ax = plt.subplots(figsize=figsize);
    
    if (saveFig):
        plt.ioff()
    else:
        plt.ion()
        
    ax.imshow(image,aspect='equal')
    
    # Draw ground truth boxes if available
    if (target != None):
        absolute_box_label = target.clone()
        if (absolute_box_label.shape[0] != 0):
            absolute_box_label = absolute_box_label[:,1:]
            absolute_box_label[:,2] = absolute_box_label[:,2] + absolute_box_label[:,0]
            absolute_box_label[:,3] = absolute_box_label[:,3] + absolute_box_label[:,1]

            for ii, box in enumerate(absolute_box_label):
                upper_left_x = box[0]
                upper_left_y = box[1]
                ww  = box[2] - box[0]
                hh  = box[3] - box[1]
                rect = patches.Rectangle(
                    (upper_left_x, upper_left_y),
                    ww, hh,
                    linewidth=5,
                    edgecolor='g',
                    facecolor="none",
                )
                ax.add_patch(rect);

    # Draw predicted absoulte boxes if available
    if (pred_absolute_boxes != None):
        confidences = pred_person_cls
        boxes       = pred_absolute_boxes 
        nms_picks   = torchvision.ops.nms(boxes, confidences, nmsIoUTreshold) 

        boxes_to_draw = boxes[nms_picks].detach().cpu().numpy()
        confs_to_draw = confidences[nms_picks].detach().cpu().numpy()

        for ii, box in enumerate(boxes_to_draw):
            if(confs_to_draw[ii] > predConfPlotTreshold):
                upper_left_x = box[0];
                upper_left_y = box[1];
                ww  = box[2] - box[0]
                hh  = box[3] - box[1]
                
                conf = "{:.3f}".format(confs_to_draw[ii])
                
                if not saveFig:
                    print(f'Conf{ii} : {confs_to_draw[ii]}')
                
                plt.text(upper_left_x,upper_left_y-5, conf, fontsize = 12,color= color)
                rect = patches.Rectangle(
                    (upper_left_x, upper_left_y),
                    ww, hh,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect);
    
    
    if saveFig:
        trainingOutpDir = os.path.join(".","Training Outputs")
        saveDir = os.path.join(trainingOutpDir,folderName)

        if not (os.path.isdir(trainingOutpDir)):
            os.mkdir(trainingOutpDir)

        if not (os.path.isdir(saveDir)):
            os.mkdir(saveDir)
            
        if (imageID == None):
            imageID = 'NA'
        else:
            imageID = str(int(imageID))
            

        imageName = folderName+"_ImgId_"+imageID+".png"
        imageDir = os.path.join(saveDir, imageName)
        plt.savefig(imageDir)
        plt.close('all')
        plt.cla()

    else:
        plt.show()
        plt.close('all')
    
    
def generateAnchorsInOrigImage(anchors,headgridSize,originalPicSize=512):
    '''
    Prepares anchor tensors in original image. 
    
    E.g. If there are 4 anchors for the prediction head, 
    4 anchor positions in original image are calculated for (x=0, y=0),(x=1, y=0)... feature grid, and written 
      one under the other to anchorsInOrig
    
    Args:
        anchors : (tuple) Tuple of anchor boxes in Tensor w,h form Tuple(Shape[numAnchors,2])
        headgridSize : Prediction head grid size, 16 or 32 for mobilenet
        originalPicSize : original image size
        
    Return:
        anchorsInOrig : Tensor shape[#ofboxes*head width size*head height size,4], anchors are written in (cx, cy, w, h) form
    '''
    scale = originalPicSize/headgridSize
    anchorsInOrig = torch.zeros([len(anchors)*headgridSize*headgridSize,4])
    numOfAnchorBox = len(anchors)
    for i in range(headgridSize):
        for j in range(headgridSize):
            for k in range(len(anchors)):
                cx = j*scale + (scale+1)/2
                cy = i*scale + (scale+1)/2
                w, h = anchors[k]
                tempAnch = torch.tensor([cx,cy,w,h])
                anchorsInOrig[i*headgridSize*numOfAnchorBox + j*numOfAnchorBox + k,:]=tempAnch
    
#     anchorsInOrig.requires_grad_(True) # does no effect result
    return anchorsInOrig 


def prepareHeadDataforLoss(HeadBB,HeadConf):
    '''
    Prepares prediction head tensors for loss calculation
    
    E.g. If there are 4 BBs for the prediction head, 
    4 BB positions in delta form are written one under the other,  for (x=0, y=0),(x=1, y=0)... of feature grid and returned
    
    Args:
        HeadBB : (tensor) Location head of the layer Shape[numofAncBoxesperCell * 4, head width, head height ]
            Boxes -> [dcx, dcy, dw, dh ]
        HeadConf : (tensor) Confidence head of the layer Shape[numofAncBoxesperCell * 2, head width, head height ]
            Confidences -> (p(person), p(background))
        
    Return:
        BBs : (tensor) Predicted bounding boxes are written in delta form (dcx, dcy, dw, dh) 
        shape[numofAncBoxesperCell * head width * head height ,4]  -> shape[4096,4] for 32x32 head
        
        CFs : (tensor) Class confidences are written in (p(person), p(background)) 
        shape[#ofPredperFeatureCell * head width * head height ,2]  -> shape[4096,2] for 32x32 head
    '''
    width = HeadBB.shape[1]
    height = HeadBB.shape[2]
    
    numOfAnchorBox = int(HeadBB.shape[0]/4)
    BBs = torch.zeros([width*height*numOfAnchorBox,4]).to(device)
    CFs = torch.zeros([width*height*numOfAnchorBox,2]).to(device)
    for i in range(width):
        for j in range(height):
            for k in range(numOfAnchorBox):
                BBs[i*height*numOfAnchorBox + j*numOfAnchorBox + k,:] = HeadBB[k*4:k*4+4,i,j]
                CFs[i*height*numOfAnchorBox + j*numOfAnchorBox + k,:] = HeadConf[k*2:k*2+2,i,j]
    
    return BBs, CFs


def prepareHeadDataforLoss_fast(HeadBB,HeadConf):
    '''
    Same function with prepareHeadDataforLoss(), but blackbox faster implementation. 
    See details in prepareHeadDataforLoss()
    '''

    BBs = HeadBB.squeeze(0)
    BBs = BBs.permute((1,2,0))
    BBs = BBs.contiguous().view(-1,4)

    CFs = HeadConf.squeeze(0)
    CFs = CFs.permute((1,2,0))
    CFs = CFs.contiguous().view(-1,2)
    return BBs, CFs


# https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def point_form(boxes):
    """ Convert box in form (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) boxes in (cx, cy, w, h) form
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

# https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4]. xmin, ymin, xmax, ymax form
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = box_a.size(0)
    B = box_b.size(0)
    box_a = box_a.to(device)
    box_b = box_b.to(device)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inter = intersect(box_a, box_b) # boxes are in the form of xmin, ymin, xmax, ymax
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    
    area_a = area_a.to(device)
    area_b = area_b.to(device)
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def collate_fn(batch):
    """
    Custom collate function.
    Need to create own collate_fn Function for COCO.
    Merges a list of samples to form a mini-batch of Tensor(s).
    Used when using batched loading from a map-style dataset.
    """
    return zip(*batch)

def sampleRandomPicsFromCOCO_old(train_loader, numtoPlot = 10, PictureSize = 512): 
    '''
    This function is used to sample random pictures from COCO dataset
    
    Args: 
    numtoPlot : number of random pictures to plot from dataset
    
    Return: 
    SelectedPics : (tensor) size[numtoPlot, 3, PictureSize, PictureSize]
    SelectedTargets: list[(tensor)] list of bounding boxes in COCO format for each picture

    '''
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    numofbatches = len(train_loader)
    batchsize = train_loader.batch_size
    randomBatches = random.sample(range(0, numofbatches), numtoPlot)

    selectedTargets = []
    selectedPics = torch.zeros((numtoPlot,3,PictureSize,PictureSize)).to(device)
    dataloader_iterator = iter(train_loader)

    
    i = 0
    batchnum = 0    
    while batchnum < numofbatches:
#         print(batchnum)
        if batchnum in randomBatches:
            data = next(dataloader_iterator)
            picnum = random.randrange(0, batchsize, 1)
            randomBatches.remove(batchnum)

            imageBatch, targetBatch, picNum = data
            image = imageBatch[picnum].unsqueeze(0).clone().to(device) 
            target = targetBatch[picnum].clone().to(device) 

            selectedPics[i,:,:,:] = image
            selectedTargets.append(target)
            i += 1
        else:
            next(dataloader_iterator)
            
        batchnum += 1 

        if not randomBatches:
            break
            
    return selectedPics, selectedTargets


def sampleRandomPicsFromCOCO(dataset, numtoPick = 10, pickSame = False): 
    '''
    This function is used to sample random pictures from a COCO type dataset
    
    Args:
    dataset: dataset to be sampled
    numtoPick : number of random pictures to pick from dataset
    pickSame: if it is set to true, 
    
    Return: 
    SelectedPics : (tensor) size[numtoPlot, 3, PictureSize, PictureSize]
    SelectedTargets: list[(tensor)] list of bounding boxes in COCO format for each picture
    '''
    
    if pickSame:
        random.seed(1234)
    else:
        pass
    
    random_indices = random.sample(range(len(dataset)), numtoPick)
    
    rand_sampler = torch.utils.data.SubsetRandomSampler(random_indices)
    loader = torch.utils.data.DataLoader(dataset,
                        sampler=rand_sampler,
                        batch_size=1,
                        collate_fn=collate_fn,
                        drop_last=False)
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    selectedTargets = []
    selectedPics = torch.zeros((numtoPick, 3, 512, 512)).to(device)
    picIds = []
    
    for i, data in enumerate(loader):
        imageBatch, targetBatch, picNum = data
        image = imageBatch[0].unsqueeze(0).to(device) 
        target = targetBatch[0].to(device) 

        selectedPics[i,:,:,:] = image
        selectedTargets.append(target)
        picIds.append(picNum[0])
    
    return selectedPics, selectedTargets, picIds


def saveOutputs(pictures, picIds, targets, preds, anchors_head1, anchors_head2,
                savefolderName='UnconstFPT',
                nmsIoUTreshold = 0.45, predConfPlotTreshold = 0.6, figsize=(8,8)):
    '''
    Saves pictures,ground truths and model predictions under specified folder
    '''
    predsPostProcess = PredsPostProcess(512, anchors_head1, anchors_head2)
    
    image_width = pictures.shape[2]
    image_height = pictures.shape[3]
    
    BBs1 = preds[0].clone()
    CFs1 = preds[1].clone()
    BBs2 = preds[2].clone()
    CFs2 = preds[3].clone()
    
    for imgNum in tqdm(range(0,pictures.shape[0])):
    
        img = pictures[imgNum,:,:,:].clone()
        target = targets[imgNum].clone()
        pred = (BBs1[imgNum,:,:,:].unsqueeze(0), CFs1[imgNum,:,:,:].unsqueeze(0), 
                BBs2[imgNum,:,:,:].unsqueeze(0), CFs2[imgNum,:,:,:].unsqueeze(0))
        id = picIds[imgNum]

        absolute_boxes,person_cls = predsPostProcess.getPredsInOriginal(pred)
    
        plot_image_mnv2_2xSSDlite(img, pred_person_cls = person_cls, pred_absolute_boxes = absolute_boxes, color = 'r'
                              ,nmsIoUTreshold = nmsIoUTreshold, predConfPlotTreshold = predConfPlotTreshold,
                              target=target, figsize=figsize,
                              saveFig=True, imageID= id, folderName = savefolderName)

class PredsPostProcess:
    '''
    Class to convert mobilenet SSD heads to real image coordinates in form [xmin, ymin, xmax, ymax]
    
    '''
    def __init__(self, image_width, anchors_head1, anchors_head2):
        Head1AnchorsForLoss = generateAnchorsInOrigImage(anchors_head1,headgridSize=32,originalPicSize=image_width)
        Head2AnchorsForLoss = generateAnchorsInOrigImage(anchors_head2,headgridSize=16,originalPicSize=image_width)
        AnchorsFlatten_wh = torch.cat((Head1AnchorsForLoss,Head2AnchorsForLoss),0) # shape[32x32x4+16x16x5, 4] 
                                                                                   # boxes in form[cx, cy, w, h]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        AnchorsFlatten_wh = AnchorsFlatten_wh.to(device)
        self.AnchorsFlatten_wh = AnchorsFlatten_wh
        self.softmax_fcn    = torch.nn.Softmax(dim=1).to(device)
        self.var_x = 0.1
        self.var_y = 0.1
        self.var_w = 0.2
        self.var_h = 0.2
    
    def getPredsInOriginal(self,preds):
        '''
            Args: 
                preds: Prediction heads, i.e output of mobilenet model()
                
            Return: 
                absolute_boxes: 32 * 32 *4  + 16 * 16 * 5 = 5376 pred BB's in form [imagenum, xmin, ymin, xmax, ymax]
                                (tensor) [5376, 5]
                person cls: Person classification heads, (tensor) [5376,1]
                
        '''
        AnchorsFlatten_wh = self.AnchorsFlatten_wh
        BBhires, CFhires = prepareHeadDataforLoss_fast(preds[0].data,preds[1].data)
        BBlores, CFlores = prepareHeadDataforLoss_fast(preds[2].data,preds[3].data)

        
        cls = torch.cat(( CFhires, CFlores))
        cls = self.softmax_fcn(cls)
        person_cls =cls[:,0]

        delta_boxes_wh = torch.cat(( BBhires, BBlores))

        pred_cx = delta_boxes_wh[:,0]*self.var_x*self.AnchorsFlatten_wh[:,2] + self.AnchorsFlatten_wh[:,0]
        pred_cy = delta_boxes_wh[:,1]*self.var_y*self.AnchorsFlatten_wh[:,3] + self.AnchorsFlatten_wh[:,1]
        pred_w = (delta_boxes_wh[:,2]*self.var_w).exp()*self.AnchorsFlatten_wh[:,2]
        pred_h = (delta_boxes_wh[:,3]*self.var_h).exp()*self.AnchorsFlatten_wh[:,3]

        absolute_xleft = pred_cx - pred_w/2
        absolute_ytop = pred_cy - pred_h/2
        absolute_xright = pred_cx + pred_w/2
        absolute_ybottom = pred_cy + pred_h/2

        absolute_boxes = torch.cat((absolute_xleft.view(-1,1), absolute_ytop.view(-1,1), absolute_xright.view(-1,1), absolute_ybottom.view(-1,1)), dim=1)

        return absolute_boxes, person_cls
    
        
def mAP(cocoGT, cocoDT, imgIDS, catIDS=1, annType="bbox"):
    """
    Explanation: This function calculate the mean average precision for given
                 ground truths and detection results. Default category and
                 annotation format is set to 'person' and 'bbox' respectively.
                 This function is based on popular benchmark function "pycocotools"
                 that is forked 3.3k. Please re-check the iou threshold (parameter iouThrs)
                 ,which is default '.5:.05:.95', before you run the code.
    Arguments:
        cocoGT(Json File): Annotated orginal valset of COCO.
        cocoDT(Json File): Model Results as format ===> [{"image_id":42, "category_id":18, "bbox":[258.15,41.29,348.26,243.78],"score":0.236}, 
                                                         {"image_id":73, "category_id":11, "bbox":[61,22.75,504,609.67],       "score":0.318}, 
                                                          ...]
        imgIDS(list): list of image IDs.
        catIDS(list): list of category ids. Default=1 as person.
        annType(String): Annotation type, Default=bbox. Can be ['segm','bbox','keypoints'].
    Returns:
        None: just results as strings in terminal.
    ######################## More Detailed Guideline ########################
    The usage for CocoEval is as follows:                                   #
       cocoGt=..., cocoDt=...       # load dataset and results              #
       E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object            #
       E.params.recThrs = ...;      # set parameters as desired             #
       E.evaluate();                # run per image evaluation              #
       E.accumulate();              # accumulate per image results          #
       E.summarize();               # display summary metrics of results    #
    #########################################################################
    The evaluation parameters are as follows (defaults in brackets):        #
       imgIds     - [all] N img ids to use for evaluation                   #
       catIds     - [all] K cat ids to use for evaluation                   #
       iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation         #
       recThrs    - [0:.01:1] R=101 recall thresholds for evaluation        #
       areaRng    - [...] A=4 object area ranges for evaluation             #
       maxDets    - [1 10 100] M=3 thresholds on max detections per image   #
       iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'   #
       iouType replaced the now DEPRECATED useSegm parameter.               #
       useCats    - [1] if true use category labels for evaluation          #
    Note: if useCats=0 category labels are ignored as in proposal scoring.  #
    Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.       #
    #########################################################################
    evaluate(): evaluates detections on every image and every category and  #
    concats the results into the "evalImgs" with fields:                    #
       dtIds      - [1xD] id for each of the D detections (dt)              #
       gtIds      - [1xG] id for each of the G ground truths (gt)           #
       dtMatches  - [TxD] matching gt id at each IoU or 0                   #
       gtMatches  - [TxG] matching dt id at each IoU or 0                   #
       dtScores   - [1xD] confidence of each dt                             #
       gtIgnore   - [1xG] ignore flag for each gt                           #
       dtIgnore   - [TxD] ignore flag for each dt at each IoU               #
    #########################################################################
    accumulate(): accumulates the per-image, per-category evaluation        #
    results in "evalImgs" into the dictionary "eval" with fields:           #
       params     - parameters used for evaluation                          #
       date       - date evaluation was performed                           #
       counts     - [T,R,K,A,M] parameter dimensions (see above)            #
       precision  - [TxRxKxAxM] precision for every evaluation setting      #
       recall     - [TxKxAxM] max recall for every evaluation setting       #
    Note: precision and recall==-1 for settings with no gt objects.         #
    #########################################################################
    ***For more details of COCOeval please check: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    ***If you need an orginal example from API please check: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    """
    
    cocoEval = COCOeval(cocoGT,cocoDT,annType)
    cocoEval.params.imgIds = imgIDS
    cocoEval.params.catIds = catIDS
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    
def round_floats(o):
    '''
    Used to round floats before writing to json form
    '''
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def get_FPnum_per_image(bbox, GT_bbox, min_IoU = 0.5):
    ''' Founds the number of False Positives by assocating detection BB's to GT BBs
    
    Arguments:
    -------------
    bbox : list
        N x 4 list of detection bounding boxes in xmin, ymin, w, h form
    GT_bbox : list
        N x 4 list of ground truth bounding boxes in xmin, ymin, w, h form
    min_IoU : float [0,1]
        Treshold of intersection of union to evaluate detection and GT to be matched, if IoU of Det and GT is below
        this value they are automatically marked as unmatched
    '''
    
    bbox = torch.tensor(bbox)

    # Convert x,y,w,h -> xmin, ymin, xmax, ymax
    bbox[:,2] = bbox[:,0] + bbox[:,2]
    bbox[:,3] = bbox[:,1] + bbox[:,3]

    GT_bbox[:,2] = GT_bbox[:,0] + GT_bbox[:,2]
    GT_bbox[:,3] = GT_bbox[:,1] + GT_bbox[:,3]

    IoUscore = jaccard(GT_bbox, bbox)

    num_det = IoUscore.shape[1]
    num_TP = 0
    GT_indexes = [x for x in range(IoUscore.shape[0])]

    # all detections
    for det_idx in range(IoUscore.shape[1]):
        
        max_IoU = min_IoU
        max_IoU_gt_id = None

        # all remained unmatched GTs
        for i, gt_idx in enumerate(GT_indexes):
            currentIoU = IoUscore[gt_idx, det_idx]
            if currentIoU > max_IoU:
                max_IoU = currentIoU
                max_IoU_gt_id = i

        if max_IoU_gt_id is not None:
            del GT_indexes[max_IoU_gt_id] # Remove GT from unmatcheds list
            num_TP += 1

        if len(GT_indexes) == 0:
            break

    FP_count_image = num_det - num_TP
    return FP_count_image


def calculatemAP(model, test_loader,cocoGT, ANCHORS_HEAD1, ANCHORS_HEAD2 , PredMinConfTreshold=0.7 , 
                 nmsIoUTreshold = 0.5, mAPOnlyFirstBatch= False, calculate_FP_ratio=False, hardware_mode = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = time.time()
    print('mAP calculation started...')
    predsPostProcess = PredsPostProcess(512, ANCHORS_HEAD1, ANCHORS_HEAD2)

    dataDictList =[]
    imgIDS = []
    model.eval()
    
    total_GT_count = 0
    total_FP_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            imageBatch, targetBatch , idxBatch = data

            imageStack = torch.stack(imageBatch).detach().to(device)
            predBatch = model(imageStack)
            
            # Outputs are in [-128, 127] in hw mode 
            if hardware_mode:
                BBs1 = predBatch[0].detach() / 128.0
                CFs1 = predBatch[1].detach() / 128.0
                BBs2 = predBatch[2].detach() / 128.0
                CFs2 = predBatch[3].detach() / 128.0
            else:
                BBs1 = predBatch[0].detach()
                CFs1 = predBatch[1].detach()
                BBs2 = predBatch[2].detach()
                CFs2 = predBatch[3].detach()

            for imgNum in range(imageStack.shape[0]):

                img = imageStack[imgNum,:,:,:]
                target = targetBatch[imgNum]
                image_id = int(idxBatch[imgNum])
                imgIDS.append(image_id)

                pred = (BBs1[imgNum,:,:,:].unsqueeze(0), CFs1[imgNum,:,:,:].unsqueeze(0), 
                        BBs2[imgNum,:,:,:].unsqueeze(0), CFs2[imgNum,:,:,:].unsqueeze(0))

                absolute_boxes, person_cls = predsPostProcess.getPredsInOriginal(pred)

                confidences = person_cls
                boxes       = absolute_boxes 
                nms_picks   = torchvision.ops.nms(boxes, confidences, nmsIoUTreshold) 
                boxes_to_draw = boxes[nms_picks]
                confs_to_draw = confidences[nms_picks]

                # Predictions filtered by nms and conf tresholding, these will go to mAP
                confMask = (confs_to_draw > PredMinConfTreshold)
                
                # Accumulate total GT bounding box number to calculate total False Positive rate 
                if calculate_FP_ratio and (target.shape[0] != 0):
                    GT_bbox = target[:,1:]
                    total_GT_count += GT_bbox.shape[0]
                    
                # Inputs to mAP algorithm
                if (confMask.any()):

                    # pred boxes -> [xmin,ymin,xmax,ymax], tensor shape[numpred,4]
                    bbox = boxes_to_draw[confMask]
                    # Convert BB to coco annot format -> [xmin,ymin,width, height]
                    bbox[:,2] = bbox[:,2] - bbox[:,0]
                    bbox[:,3] = bbox[:,3] - bbox[:,1]

                    bbox = bbox.tolist() # pred boxes -> [xmin,ymin,xmax,ymax], shape[numpred,4]
                    score = confs_to_draw[confMask].tolist()
                    category_id = np.ones_like(score,dtype=int).tolist()

                    for j in range(len(bbox)):
                        box = {"image_id":image_id, "category_id":category_id[j], "bbox":bbox[j],"score":score[j]}
                        dataDictList.append(round_floats(box))
                        
                    # If detection exists and false positive ratio calculation is enabled
                    if calculate_FP_ratio:
                        # Note that scores are already in descending order thanks to nms operation
                        # No ground truth, all detections are FP
                        if GT_bbox.shape[0] == 0:
                            total_FP_count += len(score)

                        # Find false positives
                        else:
                            FP_count_image = get_FPnum_per_image(bbox, GT_bbox, min_IoU=0.5)
                            total_FP_count += FP_count_image

            if mAPOnlyFirstBatch:
                break
                    
    if (len(dataDictList)):        
        # Evavluate and Accumulate mAP for remained baches, if any
        cocoDT = json.dumps(dataDictList)

        # Write detections to .json file
        with open('cocoDT.json', 'w') as outfile:
            outfile.write(cocoDT) 

        # Load detections
        cocoDT=cocoGT.loadRes('cocoDT.json')

        # running evaluation
        annType = 'bbox'
        cocoEval = COCOeval(cocoGT,cocoDT,annType)
        cocoEval.params.catIds = 1
        cocoEval.params.imgIds  = imgIDS
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Print False Positive Statistics
        if calculate_FP_ratio:
            print()
            print('********** False Positive Statistics **********')
            print(f'Total GT Boxes: {total_GT_count}, Total FPs Boxes: {total_FP_count}, FP% : {total_FP_count/total_GT_count*100}')
            print()
        

        mean_ap = cocoEval.stats[0].item()
        mean_recall = cocoEval.stats[8].item()

        # Delete detection json file created
        os.remove("cocoDT.json")
    else:
        mean_ap = 0
        mean_recall = 0
    t2 = time.time()
    print(f'mAP done in : {t2-t1} secs')
    return mean_ap, mean_recall


def batchNormAdaptation(model, train_loader,numSamples = 100):
    '''
    BN parameters of intel model is spoiled intentionally/or unintentionaly before publishing.
    Batch norm adaptation routine is proposed before any training based on this model.
    https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#batch-norm-statistics-adaptation
    #numSamples predictions are made and running mean variance are recalculated for the layers. 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('')
    print('Batch norm adaptation before training started.')


    for i, data in enumerate(train_loader):

        imageBatch, targetBatch, imgIDs = data

        imageStack = torch.stack(imageBatch)
        imageStack = imageStack.detach()
        imageStack.requires_grad_(False)
        imageStack = imageStack.to(device)

        predBatch = model(imageStack)
        if (i*len(imgIDs) >= numSamples):        
            return model

# Some functions to be used in training phase
def conv_model_fptunc2fpt(model):
    layer_str_arr = [attr_name for attr_name in dir(model) if
                     isinstance(getattr(model, attr_name), qat_core.layers.shallow_base_layer)]
    # Convert layers
    for layer in layer_str_arr:
        layer_attribute = getattr(model, layer)
        layer_attribute.mode_fptunconstrained2fpt('fpt')
        setattr(model, layer, layer_attribute)
    
    # Convert add_residual modules.
    add_res_attribute = getattr(model, 'add_residual')
    add_res_attribute.mode_fptunconstrained2fpt('fpt')
    setattr(model, 'add_residual', add_res_attribute)
        
    return model


def conv_model_fpt2qat(model, weight_dictionary, shift_quantile=0.985):
    print('Folding BNs and converting to qat mode')
    layer_attributes = []
    for layer_string in dir(model):
        if(layer_string in weight_dictionary):
            layer_attribute = getattr(model, layer_string)
            
            if layer_attribute.mode == 'fpt':
                print('Folding BN for:', layer_string)
                weight_bits=weight_dictionary[layer_string]
                print(f'Layer bit is : {weight_bits}')
                
                # For binary weights convert layer in to qat_ap mode
                if weight_bits == 1:
                    print('layer is converted in to qat_ap mode')
                    layer_attribute.configure_layer_base(weight_bits=2 , bias_bits=8, shift_quantile=shift_quantile)
                    layer_attribute.mode_fpt2qat('qat_ap');
                # convert other layers in to qat mode
                else:
                    print('layer is converted in to qat mode')
                    layer_attribute.configure_layer_base(weight_bits=weight_bits , bias_bits=8, shift_quantile=shift_quantile)
                    layer_attribute.mode_fpt2qat('qat');
                
                setattr(model, layer_string, layer_attribute)
                print('')
                
            else:
                print('To convert model to QAT mode, all layers must be in fpt mode but, ' + layer_string + 'is in' + layer_attribute.mode +' mode. Exiting...')
                sys.exit()

    add_res_attribute = getattr(model, 'add_residual')
    if add_res_attribute.mode == 'fpt':
        add_res_attribute.mode_fpt2qat('qat')
        setattr(model, 'add_residual', add_res_attribute)
    else:
        print('To convert model to QAT mode, add_residual modüle must be in fpt mode but, it is in ' + add_res_attribute.mode + ' mode. Exiting...')
        sys.exit()
    
    print('********* Converting to qat mode finished *********')
    print('')
    return model

def conv_model_qat2hw(model):
    print('Converting model to eval/hw mode for testing')
    
    layer_str_arr = [attr_name for attr_name in dir(model) if
                     isinstance(getattr(model, attr_name), qat_core.layers.shallow_base_layer)]
    
    for layer in layer_str_arr:
        layer_attribute = getattr(model, layer)

        if layer_attribute.mode == 'qat':
            layer_attribute.mode_qat2hw('eval')
            setattr(model, layer, layer_attribute)
#             print(f'{layer} was in qat converted to eval mode')
        elif layer_attribute.mode == 'qat_ap':
            layer_attribute.mode_qat_ap2hw('eval')
            setattr(model, layer, layer_attribute)
#             print(f'{layer} was in qat_ap converted to eval mode')
        else: 
            print('To convert model to hw mode, all layers must be in qat or qat_ap mode but, ' + layer_string + 'is in' + layer_attribute.mode +' mode. Exiting...')
            sys.exit()
#         print('')
    model  = model.to(model.conv1.op.weight.device.type)
    
    # Convert add residual operation in to eval mode
    add_res_attribute = getattr(model, 'add_residual')
    if add_res_attribute.mode == 'qat':
        add_res_attribute.mode_qat2hw('eval')
        setattr(model, 'add_residual', add_res_attribute)
    else:
        print('To convert model to QAT mode, add_residual modüle must be in qat mode but, it is in ' + add_res_attribute.mode + ' mode. Exiting...')
        sys.exit()
    
    print('********* Converting model to eval/hw mode for testing finished *********')
    print('')
    return model