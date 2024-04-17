import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (Profile, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class Identify:
    def __init__(self):
        self.cap = cv2.VideoCapture()
        # 模型相关
        self.people_weights = ROOT / 'weights/people.pt'  # 人的权重模型
        self.mask_weights = ROOT / 'weights/mask.pt'    # 口罩的权重模型
        self.conf_thres = 0.25  # 识别置信度阈值
        self.iou_thres = 0.25  # 识别非极大抑制时的 IoU 阈值
        self.img_size = 640  # 预测时网络输入图片的尺寸，默认值为 [640]
        self.classes = 0  # 指定要检测的目标类别，默认为None，表示检测所有类别
        self.max_det = 1000  # 每张图像的最大检测框数，默认为1000。
        self.agnostic_nms = False  # 是否使用类别无关的非极大值抑制，默认为False
        self.line_thickness = 3  # 检测框的线条宽度，默认为3
        self.device = ''  # 使用的设备，可以是CUDA设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
        self.augment = False  # 是否使用数据增强进行推理
        self.visualize = False  # 是否可视化模型中的特征图，默认为False
        self.dnn = False  # 是否使用 OpenCV DNN 进行 ONNX 推理
        self.half = False  # 是否使用 FP16 半精度进行推理
        self.class_nums = 0  # 识别的类型个数
        self.load_people_model()  # 加载人的检测模型
        self.load_mask_model()  # 加载口罩的检测模型
        """====================================加载模型========================================"""
    def load_people_model(self):
        device = select_device(self.device)
        self.people_model = DetectMultiBackend(self.people_weights, device=device, dnn=self.dnn, data='', fp16=self.half)
        self.people_stride, self.people_names, self.people_pt = self.people_model.stride, self.people_model.names, self.people_model.pt
        self.people_class_nums = len(self.people_names)  # 获取识别的个数

    def load_mask_model(self):
        device = select_device(self.device)
        self.mask_model = DetectMultiBackend(self.mask_weights, device=device, dnn=self.dnn, data='', fp16=self.half)
        self.mask_stride, self.mask_names, self.mask_pt = self.mask_model.stride, self.mask_model.names, self.mask_model.pt
        self.mask_class_nums = len(self.mask_names)  # 获取识别的个数

    # 在人的检测框中检测口罩
    def detect_masks_in_people(self, image, people_detections):
        # 存储口罩检测结果的列表
        mask_labels = []
        # 存储口罩检测框的坐标
        mask_boxes = []
        # 对于每个人的检测结果，进行口罩检测
        for person_detection in people_detections:
            person_bbox = person_detection[:4]  # 提取人的边界框坐标
            person_img = image[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])]

            if person_img.shape[0] == 0 or person_img.shape[1] == 0:
                # 处理图像大小为零的情况
                continue

            # 图像预处理
            img = letterbox(person_img, self.img_size, stride=self.mask_stride, auto=self.mask_pt)[
                0]  # padded resize
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)  # contiguous

            with torch.no_grad():
                img = torch.from_numpy(img).to(self.mask_model.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim

            # Inference
            pred = self.mask_model(img, augment=self.augment, visualize=self.visualize)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                       agnostic=self.agnostic_nms)

            # 处理口罩检测结果
            if pred is not None and len(pred):
                for det in pred:
                    for *xyxy, conf, cls in det:
                        label = f'Mask {conf:.2f}'
                        mask_labels.append(label)  # 添加口罩检测结果到标签列表

                        # 计算口罩框相对于人的边界框的位置
                        mask_x1 = max(int(xyxy[0]) + int(person_bbox[0]), 0)
                        mask_y1 = max(int(xyxy[1]) + int(person_bbox[1]), 0)
                        mask_x2 = min(int(xyxy[2]) + int(person_bbox[0]), image.shape[1])
                        mask_y2 = min(int(xyxy[3]) + int(person_bbox[1]), image.shape[0])
                        mask_boxes.append([mask_x1, mask_y1, mask_x2, mask_y2])  # 添加口罩检测框的坐标

        return mask_labels, mask_boxes

    def show_frame(self, image, cap_flag):
        if cap_flag:
            flag, image = self.cap.read()
        if image is not None:
            img = image.copy()
            show_img = img
            labels = []
            # 这里开始进行人的检测
            with torch.no_grad():
                # 图像预处理
                img = letterbox(img, self.img_size, stride=self.people_stride, auto=self.people_pt)[0]  # padded resize
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)  # contiguous
                dt = (Profile(), Profile(), Profile())
                with dt[0]:
                    img = torch.from_numpy(img).to(self.people_model.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if len(img.shape) == 3:
                        img = img[None]  # expand for batch dim
                # Inference
                with dt[1]:
                    pred = self.people_model(img, augment=self.augment, visualize=self.visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                               self.classes, self.agnostic_nms, max_det=self.max_det)
                    # Process detectionss
                # 对人的检测结果进行处理
                annotator = Annotator(show_img, line_width=self.line_thickness, example=str(self.people_names))
                people_detections = []
                if pred is not None:
                    for det in pred:
                        if det is not None and len(det):
                            for *xyxy, conf, cls in det:
                                # 添加检测框的坐标到人的检测结果列表中
                                people_detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                for i, det in enumerate(pred):  # detections per image

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], show_img.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):

                            # cv2.destroyAllWindows()
                            c = int(cls)  # integer class
                            label = f'{self.people_names[c]} {conf:.2f}'
                            labels.append(self.people_names[c])  # 获取识别类别

                            # 在人的检测框中检测口罩
                            person_img = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            mask_labels, mask_boxes = self.detect_masks_in_people(person_img, people_detections)

                            # 绘制口罩框
                            for mask_box in mask_boxes:
                                annotator.box_label(mask_box, "Mask", color=(0, 255, 0))  # 使用绿色表示口罩

                            annotator.box_label(xyxy, label, color=(0, 0, 255))
                show_img = annotator.result()
            return image, show_img, labels
        else:
            return image, None, None
