## Dataset setup

### Setup from original source 
You can obtain the Human3.6M dataset from the [Human3.6M](http://vision.imar.ro/human3.6m/) website, and then set it up using the instructions provided in [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). 

### Setup from preprocessed dataset (Recommended)
 You also can access the processed data by downloading it from [here](https://drive.google.com/drive/folders/1EqQ4x_Ldrra9tztjv4qO_w2rkNzTuUM8?usp=sharing).

```python
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```



## Train the model from scratch
The log file, pre-trained model, and other files of each training time will be saved in the './checkpoint' folder.

For Human3.6M:

```bash
python main.py --train --model model_IGANet --layers 3 --nepoch 20 --gpu 0
```

## Test the pre-trained model
The pre-trained model can be found [here](https://drive.google.com/drive/folders/1EqQ4x_Ldrra9tztjv4qO_w2rkNzTuUM8?usp=sharing). please download it and put it in the 'args.previous_dir' ('./pre_trained_model')
directory.


To Test the pre-trained model on Human3.6M:
```bash
python main.py --reload --previous_dir "./pre_trained_model" --model model_IGANet --layers 3 --gpu 0
```



## Demo 

This visualization code is designed for single-frame based models, making it easy for you to perform 3D human pose estimation on a single image or video.


Before starting, Prerequisites:

- Download YOLOv3 and HRNet pretrained models from [here](https://drive.google.com/drive/folders/1EqQ4x_Ldrra9tztjv4qO_w2rkNzTuUM8?usp=sharing) and place them in the './demo/lib/checkpoint' directory. 

- Copy your image (or video) to the './demo/images (or videos)' directory.

- Make sure to place the pre-trained model in the 'args.previous_dir' ('./pre_trained_model') directory.


testing on images:

```bash
python demo/vis.py --type 'image' --path './demo/images/shukla.JPG' --gpu 0
```

<p align="center"><img src="images/shukla_2d.png" width="35%" alt="" />
<img src="images/shukla_3d.png" width="55%" alt="" /></p>

testing on videos:
```bash
python demo/vis.py --type 'video' --path './demo/videos/jash_walking_1.mp4' --gpu 0
```

<p align="center"><img src="images/jash_walking_1.gif" width="85%" alt="" /></p>