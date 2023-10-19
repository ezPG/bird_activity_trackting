import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

class CustomFasterRCNNDataset(Dataset):
    def __init__(self, image_folder, label_folder, transforms = None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = os.listdir(self.image_folder)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.image_files[idx].replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.webp', '.xml'))
        # print(f"Image path: {image_path}")
        # print(f"Label path: {label_path}")

        image = Image.open(image_path).convert("RGB")
        annotation = self.parse_xml(label_path, idx)

        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)
        
        return image, annotation

    def parse_xml(self, label_path, idx):
        tree = ET.parse(label_path)
        root = tree.getroot()

        annotation = {}

        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        annotation['area'] = torch.tensor([width * height])

        # Get bounding box coordinates
        object_elem = root.find('object')
        bndbox = object_elem.find('bndbox')



        xmin = abs(int(bndbox.find('xmin').text))
        ymin = abs(int(bndbox.find('ymin').text))
        xmax = abs(int(bndbox.find('xmax').text))
        ymax = abs(int(bndbox.find('ymax').text))
        
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax += 1

        annotation['boxes'] = torch.tensor([[xmin, ymin, xmax, ymax]], dtype = torch.float32)

        # Get label
        label = object_elem.find('name').text

        if label == 'Swimming': target = 0
        elif label == 'Flying': target = 1
        elif label == 'Walking': target = 2
        else: target = 3

        annotation['labels'] = torch.tensor([target], dtype= torch.int64)
        annotation['iscrowd'] = torch.tensor(0, dtype= torch.uint8)
        annotation['image_id'] = torch.tensor([idx])

        return annotation


# dataset = CustomFasterRCNNDataset('/Users/prashantgautam/Desktop/val_split/images', '/Users/prashantgautam/Desktop/val_split/labels')

# img, anno = dataset.__getitem__(0)

# print(np.array(img).shape)
# print(anno)
