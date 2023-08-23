###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import os, sys, random, torch, torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import torchvision.ops as ops
import torch.utils.data
import numpy as np
import pandas as pd
import copy
from PIL import Image
import os.path
import time, json
from typing import Any, Callable, Optional, Tuple, List
from typing import Callable


class input_fxpt_normalize:
    def __init__(self, act_8b_mode):
        self.act_8b_mode = act_8b_mode

    def __call__(self, img):
        if(self.act_8b_mode):
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)


### Emre Can: Our COCO Dataloder for training classes at specific ratio in every batch.
def class_lookup(cls):
    c = list(cls.__bases__)
    for base in c:
        c.extend(class_lookup(base))
    return c

# ref: https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html
class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        
        annFile (string): Path to json annotation file.
        
        scaleImgforCrop (int, optional): Img and target BBs are scaled with 
        constant aspect ratio st:
            if image width, image height > scaleImgforCrop image is shrinked 
            until width or height becomes equal to scaleImgforCrop
    
            if image width, image height < scaleImgforCrop image is expanded 
            until width or height becomes equal to scaleImgforCrop
            
            else no scaling
        fit_full_img: If it is set to true, image is scaled t fully fit in the window specified by "scaleImgforCrop x scaleImgforCrop"  
        transform (callable, optional): A function/transform that  takes in an 
        PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
        
        target_transform (callable, optional): A function/transform that takes in 
        the target and transforms it.
        transforms (callable, optional): A function/transform that takes input 
        sample and its target as entry and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        scaleImgforCrop: int= None,
        fit_full_img = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.annFilePath = os.path.join('.',annFile)
        self.catPersonId = self.coco.getCatIds(catNms=['person'])[0]
        self.scaleImgforCrop = scaleImgforCrop
        self.fit_full_img = fit_full_img
        

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id, iscrowd=False))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        id = self.ids[index]
        imgID = id

        try:
            image = self._load_image(id)
        except:
            print(f'********Unable to load image with id: {imgID}********')
            print('Please check if image is corrupted, and remove it from annotations if necessary.')
            
            
        target = copy.deepcopy(self._load_target(id)) # deepcopy target list beforecentercrop manip, to be abe to work with same
                                                      # dateset without reloading it 
        
        image_width = image.size[0]
        image_height = image.size[1]
        
        
        # If necesary rescale the image and BBs near the size of planned center crop as much as possible
        scale = self._calcPrescale(image_width=image_width, image_height=image_height)
        image = self._prescaleImage(image, scale)
        
        for i, t in enumerate(target):
            BB = t['bbox'].copy()
            scaledBB = self._prescaleBB(BB,scale)
            target[i]['bbox'] = scaledBB
            
        
        
        # Image width height after prescaling
        image_width = image.size[0]
        image_height = image.size[1]
        
        # Check if center crop applied
        centerCropped = False
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
            # If center crop applied, transform BBs as well
            for t in self.transforms.transform.transforms:
                if (type(t) == torchvision.transforms.transforms.CenterCrop):
                    centerCropped = True
        
        

        x_scale = image.size(2) / image_width
        y_scale = image.size(1) / image_height

        bbox_arr = []
        
        for idx,ann in enumerate(target):
            if ann['category_id'] == self.catPersonId:
                crop_size = image.shape[1]
                
                if centerCropped:
                    bbox = ann['bbox'].copy()
                    croppedBB = self.cropBBox(bbox, crop_size,image_height,image_width)    
                else:
                    croppedBB = torch.tensor(ann['bbox']) 
                
                if not (croppedBB == None):
                    bbox_arr.append(croppedBB)                
                             
        if len(bbox_arr) != 0:
            bbox_arr = torch.stack(bbox_arr)
            wh = bbox_arr[:, 2:] 
            xy = bbox_arr[:, :2]
            
            id_tensor = torch.tensor([id]).unsqueeze(0).expand(bbox_arr.size(0), -1)

            bbox_arr = torch.cat([id_tensor, xy, wh], dim=-1)
        else:
            bbox_arr = torch.tensor(bbox_arr)
               
        return image, bbox_arr , imgID 

    def __len__(self) -> int:
        return len(self.ids)
    
    def get_labels(self):
        labels = []
        for id in self.ids:
            anns = self._load_target(id)
            person_flag = False
            for ann in anns:     
                person_flag = ann['category_id'] == self.catPersonId
                if person_flag == True:
                    break
            if person_flag == True:
                labels.append(1)
            else:
                labels.append(0)
        return torch.tensor(labels)
    
    def get_cat_person_id(self):
        return self.catPersonId
    
    def get_coco_api(self):
        return self.coco
    
    
    # Functions defined for prescaling images/targets before center crop operation
    def _calcPrescale(self, image_width, image_height):
        # Calculate scale factor to shrink/expand image to coincide width or height to croppig area
        scale = 1.0
        if self.scaleImgforCrop != None:
            if self.fit_full_img:
                max_size = max(image_width, image_height)
                scale = max_size/self.scaleImgforCrop
            else:
                # image fully encapsulates cropping area or vice versa
                if ((image_width-self.scaleImgforCrop)*(image_height-self.scaleImgforCrop) > 0): 
                    # if width of original image is closer to crop area
                    if abs(1-image_width/self.scaleImgforCrop) < abs(1-image_height/self.scaleImgforCrop):
                        scale = image_width/self.scaleImgforCrop
                    else:
                        scale = image_height/self.scaleImgforCrop
        return scale
    
    # Scales the image with defined scale
    def _prescaleImage(self, image, scale):
        image_width = int(image.size[0]/scale)
        image_height = int(image.size[1]/scale)

        t = transforms.Resize([image_height,image_width])
        image = t(image)
        return image
    
    # Scales the targets with defined scale
    def _prescaleBB(self, BB, scale):
        scaledbb = [round(p/scale,1) for p in BB]
        return scaledbb
        
    
    def cropBBox(self,bbox,crop_size, image_height, image_width):

        bbox_aligned = []
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # Casses for cropping       
        if image_height < crop_size:
            offset = (crop_size - image_height) // 2
            y = y + offset
            if (y+h) > crop_size:
                offset = (y+h)-crop_size
                h = h - offset
        if image_width < crop_size:
            offset = (crop_size - image_width) // 2
            x = x + offset
            if (x+w) > crop_size:
                offset = (x+w)-crop_size
                w = w - offset
        if image_width > crop_size:
            offset = (image_width - crop_size) // 2
            if offset > x:
                # Deal with BB coincide with left cropping boundary
                w = w -(offset-x)
                x = 0
            else:
                x = x - offset
                
                # Deal with BB coincide with right cropping boundary
                if (x+w) > crop_size:
                    offset = (x+w)-crop_size
                    w = w - offset
                    
        if image_height > crop_size:

            offset = (image_height - crop_size) // 2
            if offset > y:
                # Deal with BB coincide with top cropping boundary
                h = h -(offset-y)
                y = 0
            else:
                y = y - offset
                # Deal with BB coincide with bottom cropping boundary
                if (y+h) > crop_size:
                    offset = (y+h)-crop_size
                    h = h - offset
        
        bbox_aligned.append(x)
        bbox_aligned.append(y)
        bbox_aligned.append(w)
        bbox_aligned.append(h)

        if ((w <= 0) or (h <= 0)):
            return None
        else:
            x_scale, y_scale = 1.0,1.0
            return torch.mul(torch.tensor(bbox_aligned), torch.tensor([x_scale, y_scale, x_scale, y_scale]))
    
    def __round_floats(self,o):
        '''
        Used to round floats before writing to json file
        '''
        if isinstance(o, float):
            return round(o, 2)
        if isinstance(o, dict):
            return {k: self.__round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.__round_floats(x) for x in o]
        return o
    
    def _check_if_annot_ignored(self, annot_bbox, ignore_bboxes):
        '''gets an annotation and ignore bboxes list in [xmin, ymin, w, h] form and calculates the percentage
           of the overlapping area. If overlapping area exceeds 50% for any ignore part, returns True, otherwise returns False
        '''
        annot_bbox = annot_bbox.copy()
        annot_area = max(annot_bbox[2] * annot_bbox[3], 0)
        annot_bbox[2] = annot_bbox[0] + annot_bbox[2]
        annot_bbox[3] = annot_bbox[1] + annot_bbox[3]
        
        for ignore_bbox in ignore_bboxes:
            
            ignore_bbox  = ignore_bbox.copy()

            ignore_bbox[2] = ignore_bbox[0] + ignore_bbox[2]
            ignore_bbox[3] = ignore_bbox[1] + ignore_bbox[3]

            x_min_intersect = max(annot_bbox[0], ignore_bbox[0])
            y_min_intersect = max(annot_bbox[1], ignore_bbox[1])
            x_max_intersect = min(annot_bbox[2], ignore_bbox[2])
            y_max_intersect = min(annot_bbox[3], ignore_bbox[3])
            w = max(x_max_intersect - x_min_intersect, 0)
            h = max(y_max_intersect - y_min_intersect, 0)
            
            if annot_area <= 0:
                return True
            
            if w * h / annot_area > 0.5:
                return True
            
        return False
        
    
    def createResizedAnnotJson(self,targetFileName,cropsize=512, mask_ignore_parts=False, ignore_parts_file=None):
        '''
        Resizes person annotations after center crop operation and saves as json file to the
        directory of original annotations with the name "targetFileName"
        
        If 'mask_ignore_parts' flag set to true and corresponding wider dataset ignore_parts_file supplied, 
        annotations having 50% or more overlap with an ignore part are deleted.
        
        '''
        
        # Get ignore part bb's in to a dictionary, wit image names as keys
        if mask_ignore_parts:
            ignore_part_dict = {}
            with open(ignore_parts_file) as f:
                for t, ignore_raw in enumerate(f):
                    ignore_raw = ignore_raw.split()
                    imgName = ignore_raw[:1][0]

                    BBs_str = ignore_raw[1:]
                    bb_raw = [int(bb) for bb in BBs_str]

                    BBs = []
                    bb = []
                    for i, p in enumerate(bb_raw):
                        bb.append(p)
                        if ((i+1)%4 == 0):

                            BBs.append(bb)
                            bb = []

                    ignore_part_dict[imgName] = BBs
            
        
        t1 = time.time()
        # Get original json annot file path, and create pah for resized json annot file
        path, annotfilename = os.path.split(self.annFilePath)
        resizedAnnotPath = os.path.join(path,targetFileName)
        
        print('')
        print(f'Creating Json file for resized annotations: {resizedAnnotPath}')
        

        # Load original annotation json file as dictionary and assign it to resized annot dict
        with open(self.annFilePath) as json_file:
            resizedanotDict = json.load(json_file)

        # Original annotations array
        origannList = resizedanotDict['annotations']
        
        # Check if center crop applied
        centerCropped = False
        if self.transforms is not None:
            # If center crop applied, transform BBs as well
            for t in self.transforms.transform.transforms:
                if (type(t) == torchvision.transforms.transforms.CenterCrop):
                    centerCropped = True
                    
        
        resizedannList = []
        for resizedannot in origannList:

            currentcatID = resizedannot['category_id']
            currentBB = resizedannot['bbox']
            currentImgID = resizedannot['image_id']
            
            # if annotations overlaps with an ignore part, do not add it to new annot file
            if mask_ignore_parts:
                image_name = self.coco.loadImgs(currentImgID)[0]['file_name']
                if image_name in ignore_part_dict:
                    ignoreBBs = ignore_part_dict[image_name]
                    is_ignored = False
                    is_ignored = self._check_if_annot_ignored(resizedannot['bbox'].copy(), ignoreBBs)

                    if is_ignored:
                        continue
            
            # Get crop size and original image sizes
            image_width = self.coco.loadImgs(currentImgID)[0]['width']
            image_height = self.coco.loadImgs(currentImgID)[0]['height']

            
            # If presclae applied to image, calculate new image width and height
            scale = self._calcPrescale(image_width=image_width, image_height=image_height)
            image_width = image_width / scale
            image_height = image_height / scale

            if currentcatID == self.catPersonId:
                # if BB is person
                bbox = resizedannot['bbox'].copy()
                
                # If prescale appied to image, resize annotations BBs
                bbox = self._prescaleBB(bbox, scale)
                
                # If center crop  applied, crop/recalculate BBs as well
                if centerCropped:
                    croppedBB = self.cropBBox(bbox, cropsize,image_height,image_width)    
                else:
                    croppedBB = torch.tensor(bbox) 
                
                if (croppedBB != None):
                    # If BB is person and valid after crop, add it to resized annotations list
                    croppedBB = croppedBB.tolist()
                    resizedannot['bbox'] = self.__round_floats(croppedBB)
                    resizedannot['area'] = self.__round_floats(croppedBB[2]*croppedBB[3])
                    resizedannList.append(resizedannot)
            else:
                # If BB is non-person add it to resized annotations list as it is
                resizedannList.append(resizedannot)
                
        # If prescale or center-crop applied
        # Change width and height information of "images" field in annotations file
        origImgList = resizedanotDict['images']
        
        for i, imagInfo in enumerate(origImgList):
            curInfo = origImgList[i]
            image_width = curInfo['width']
            image_height = curInfo['height']
            
            if centerCropped:
                curInfo['width'] = cropsize
                curInfo['height'] = cropsize
            else:
                scale = self._calcPrescale(image_width=image_width, image_height=image_height)
                curInfo['width'] = int(image_width / scale)
                curInfo['height'] = int(image_height / scale)

            origImgList[i] = curInfo.copy()
                
        resizedanotDict['images'] = origImgList
        resizedanotDict['annotations'] = resizedannList
        print('Saving resized annotations to json file...')

        # Save resized annotations in json file
        resizedanotDict = json.dumps(resizedanotDict)
        with open(resizedAnnotPath, 'w') as outfile:
            outfile.write(resizedanotDict)

        print(f'{resizedAnnotPath} saved.')
        t2 = time.time()
        print(f'Elapsed time: {t2-t1} seconds')
            
# ref: https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        constantSeed: Make it true if you want same random at each run
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset,constantSeed: bool = False, indices: list = None, num_samples: int = None, 
                 callback_get_label: Callable = None, ratio: int = 4):
        # if indices is not provided, all elements in the dataset will be considered
        self.constantSeed = constantSeed
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        label_to_count[1] = int(label_to_count[1] / ratio)

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        return dataset.get_labels()

    def __iter__(self):
        if self.constantSeed:
            torch.random.manual_seed(1234)
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples