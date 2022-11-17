import sys
import torch

import time
from pathlib import Path

import cv2

import torch.backends.cudnn as cudnn

from visual_clues.utils.general import non_max_suppression

# from .tracker_model import TrackerModel
# from tracker.common.config import TRACKER_CONF

# import tracker.autotracker as at
import os

from PIL import Image

from visual_clues.utils.torch_utils import select_device
from visual_clues.models.experimental import attempt_load
from visual_clues.utils.datasets import letterbox
import random
import requests
import numpy as np
from visual_clues.utils.general import scale_coords
import os.path

class YoloTrackerModel(): # Inherits from TrackerModel ?

    def __init__(self):
        # super().__init__()
        # self.config = TRACKER_CONF()
        print("Initializing YoloV7 model.")
        self.img_size = 640
        self.stride = 32
        self.orig_img = None
        self.weights_path = '/inputs/yolov7-checkpoint/yolov7.pt'
        self.model, self.device, self.half, self.names, self.colors = self.load_model()

    
    def load_model(self):
        # self.confidence = self.config.CONFIDENCE_THRESHOLD
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        half = device.type != 'cpu'
        # checkpoints_name = 'yolov7.pt'
        # cur_dir_path = os.path.dirname(os.path.abspath(__file__))
        # weights = os.path.join(cur_dir_path, checkpoints_name)
        # if not os.path.isfile(weights):
        #     synset_url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'
        #     os.system('wget ' + synset_url)

        #"/notebooks/yolov7/yolov7.pt" #
        model = attempt_load(self.weights_path, map_location=device)  # load
        #torch.load(weights, map_location=device)  # load FP32 model

        # Convert model to FP16 (faster inference time if GPU is available)
        if half:
            model.half()

        # Get class names
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference once
        if device.type != 'cpu':
            model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))

        return model, device, half, names, colors

    def forward(self, image : Image, metadata=None):
        processed_img = self.process_frame(image)
        output = self.detect(processed_img)
        return output

    def save(self, label):
        pass

    def process_frame(self, img : Image):
        """
        Input: frame
        Output: Normalized frame
        """
        self.orig_img = img
        # Padded resize
        img = letterbox(img, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  
        return img

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def detect(self, img) -> str:
        """
        Input: processed frame
        Output: concatenated string of: class_name, bounding box, confidence
        """
        
        # Predict
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred)

        outputs = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.orig_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                class_name = self.names[int(cls)]
                bbox = str(torch.tensor(xyxy).view(1, 4).view(-1).tolist())
                confidence = str(conf.tolist())
                line = ' '.join((class_name, bbox, confidence))
                print(f"Detected: {line}")
                outputs.append({'detections_boxes': bbox, 'detection_scores': confidence, 'detection_classes': class_name})
        
        # PLOT BBOXES ON IMAGE - SANITY TEST - TO DELETE LATER
        # label = f'{self.names[int(cls)]} {conf:.2f}'
        # url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
        # # img_orig = Image.open(requests.get(url, stream=True).raw) 
        # resp = requests.get(url, stream=True).raw
        # image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # img_orig = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # self.plot_one_box(xyxy, img_orig, label=label, color=self.colors[int(cls)], line_thickness=3)
        # # TO DELETE - SANITY TEXT OF BBOXES ON IMAGE
        # cv2.imwrite("/notebooks/nebula3_videoprocessing/videoprocessing/test123.jpg", img_orig)
        # print(f" The image with the result is saved")

        return outputs


def main():
    yolo_model = YoloTrackerModel()
    # url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
    image = cv2.imread('/notebooks/yolov7/inference/images/demo.jpg')
    output = yolo_model.forward(image)
    print(output)

if __name__ == '__main__':
    main()
