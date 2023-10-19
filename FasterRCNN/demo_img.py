import torch
import cv2
import torch
import torchvision
import numpy as np
import argparse
from model import get_model
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to the output video")
args = vars(ap.parse_args())

images = [image.to(device) for image in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Make predictions with model
model = get_model().to(device)
model.load_state_dict(torch.load('faster_rcnn_model.pth'))

model.eval()
with torch.inference_mode():
    outputs = model(images)

# Get predicted bounding boxes and object classes
pred_boxes = outputs[0]['boxes'].cpu().numpy()
pred_classes = outputs[0]['labels'].cpu().numpy()