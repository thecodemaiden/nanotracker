# Performance Benchmark of Quantized Detection Model

This directory is built for comparison of our "quantized / quantization aware trained" detection algorithm to one of the SOTA compact detection algorithms, EfficientDet-d0, which has comparable complexity and structure with our quantized model.

Our person tracking algorithm uses MobileNet-v2 as backbone mechanism and combines it with 2 SSD heads using total of 9 anchor boxes. Overall model consists of 60 convolution layers. 

We quantized the layers of this model and applied "quantization aware training" methods to recover its accuracy drop due to quantization of layers and output clamping. We have re-scaled and center cropped the images in [Wider Person Dataset](https://competitions.codalab.org/competitions/20132#learn_the_details), also we resized its annotations and converted in to COCO annotation format to use them in our training/evaluation tasks. Then we applied smart training approaches which consider the effects of quantization and output clamping of the layers during optimization, which we call "quantization aware training".

Our main motivation of quantizing networks and applying quantization aware training methods is to reduce the overall network size, inference time and training effort while keeping accuracy drop in an acceptable level. We aim to develop quantized compact detection algorithms executable on low power and low cost accelerator chips.

## Dependencies
* [PyTorch](https://github.com/pytorch/pytorch)
* [Torchvision](https://github.com/pytorch/vision)
* [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
* [webcolors](https://pypi.org/project/webcolors/)
* [PyYAML](https://github.com/yaml/pyyaml)

## Evaluating EfficientDet with Wider Person Validation Dataset
In this section, steps to reproduce the evaluation of EfficientDet model from [Yet-Another-EfficientDet-Pytorch Repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git) with d0 coefficients is explained. For evaluation, aforementioned Wider Person Validation Dataset in COCO format is used. 

### 1. Clone EfficientDet to Your Local
Open a terminal and go to directory in your local where you want to clone , then type:
```bash
git clone --depth 1 https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
```

### 2. Prepare EfficientDet-d0 Coefficients
* Go to main directory
    ```bash
    cd Yet-Another-EfficientDet-Pytorch/
    ```
* Create weights folder
    ```bash
    mkdir weights
    ```
* Download EfficientDet d0 coefficients 
    ```bash
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth
    ```

### 3. Prepare Wider Person Dataset
* Download original Wider Person Dataset
    * Sign up [Codalab](https://competitions.codalab.org/) and  participate to [WIDER Face & Person Challenge 2019](https://competitions.codalab.org/competitions/20132)
    * Under "Participate" tab click "Train & Validation Data in Google Drive" and download
        * val_data.tar.gz
        * Annotations/val_bbox.txt
    * Extract val_data.tar.gz as val_data and move val_data folder under ./data/original_wider/val_data
    * Move "val_bbox.txt" under ./data/original_wider/

* Move our "wider2coco.py" script in "efficientdet_comparison" folder to main folder of your local "Yet-Another-EfficientDet-Pytorch" repository. Following code will produce resized images and annotations.
    ```bash
    python wider2coco.py -ip ./data/original_wider/val_data -af ./data/original_wider/val_bbox.txt 
    ```
* Script will automatically convert Wider Dataset in to COCO format and create following repository structure:

        ./Yet-Another-EfficientDet-Pytorch/datasets/wider/val
             image001.jpg
             image002.jpg
             ...
        ./Yet-Another-EfficientDet-Pytorch/datasets/wider/annotations
            instances_val.json
    
   

### 4. Manually Set Project's Specific Parameters

* Create a yml file "wider.yml" under "projects"
    ```bash
    touch projects/wider.yml
    ```

 * Copy following content in to "wider.yml" file
 
       project_name: wider
       train_set: train
       val_set: val
       num_gpus: 1  # 0 means using cpu, 1-N means using gpus 
        
       # Wider validation dataset mean and std in RGB order
       mean: [0.416, 0.413, 0.406]
       std: [0.308, 0.306, 0.310]
        
       # this is coco anchors, change it if necessary
       anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
       anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
        
       # objects from all labels from your dataset with the order from your annotations.
       # its index must match your dataset's category_id.
       # category_id is one_indexed,
       # for example, index of 'car' here is 2, while category_id of is 3
       obj_list: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase', 'scissors','teddy bear', 'hair drier', 'toothbrush']

### 5. Evaluate EfficientDet model performance
* Setup "person only evaluation"
    * Open "coco_eval.py" under <parent directory>/Yet-Another-EfficientDet-Pytorch
    * Paste following code after line 132, "coco_eval.params.imgIds = image_ids", to evaluate mAP results only for person category
        ```python
        coco_eval.params.catIds = 1
        ```

* For evaluation on cuda enabled platform 
    ```bash
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth
    ```

* For evaluation on cuda disabled platform 
    ```bash
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth --cuda False
    ```


## Evaluating Our Quantized MobilenetSSDLite model with Wider Person Validation Dataset

### 1. Clone Quantized Mobilenet Model to Your Local
Open a terminal and go to directory in your local where you want to clone our [Quantization Aware Training - Person Tracking](https://github.com/sai-tr/persontracking_qat.git) repository, then type:
```bash
git clone --depth 1 https://github.com/sai-tr/persontracking_qat.git
```

### 2. Prepare Wider Person Dataset
* Download original Wider Person Dataset
    * Sign up [Codalab](https://competitions.codalab.org/) and  participate to [WIDER Face & Person Challenge 2019](https://competitions.codalab.org/competitions/20132)
    * Under "Participate" tab click "Train & Validation Data in Google Drive" and download
        * val_data.tar.gz
        * Annotations/val_bbox.txt
    * Extract val_data.tar.gz as val_data and move val_data folder under ./data/original_wider/val_data
    * Move "val_bbox.txt" under ./data/original_wider/

* Move our "wider2coco.py" script in "efficientdet_comparison" folder to main folder of your local "persontracking_qat" repository. Following code will produce resized images and annotations.
    ```bash
    python wider2coco.py -ip ./data/original_wider/val_data -af ./data/original_wider/val_bbox.txt 
    ```
* Script will automatically convert Wider Dataset in to COCO format and create following repository structure:

        ./persontracking_qat/datasets/wider/val
             image001.jpg
             image002.jpg
             ...
        ./persontracking_qat/datasets/wider/annotations
            instances_val.json

### 3. Evaluate Quantized Mobilenet Model Performance
Note that model mode should match with the loaded model parameter dictionary. Selectable model modes are: 
* Full Precision Unconstrained(fpt_unc): All layers are in full precision and no output clamping
* Full Precision Constrained(fpt): All layers are in full precision and layer output are clamped to +-1 
* Quantized(qat): All layers are quantized layer outputs are clamped to +-1


* Move our "coco_eval.py" script in "efficientdet_comparison" folder to "persontracking_qat" folder and use following command for evaluation:
    ```bash
    python coco_eval.py -m qat -dp ./datasets/wider/val -ap ./datasets/wider/annotations/all_val_prep.json -wp ./efficientdet_comparison/training_experiment_best.pth.tar
    ```
    Note that: Code evaluates quantized model with weights "training_experiment_best.pth.tar", using images and annotations in paths  "./datasets/wider/val" "./datasets/wider/annotations/instances_val.json" respectively.

## mAP Comparisons
### EfficientDet-d0
    ### Wider Validation Dataset mAP scores ###
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.543
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.275
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.532
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.369
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.546
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
    

### Quantized Mobilenet
    ### Wider Validation Dataset mAP scores ###
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.281
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.107
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.324
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.637
    