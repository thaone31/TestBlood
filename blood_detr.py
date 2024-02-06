from ultralytics import YOLO 
from PyQt5.QtCore import QThread,pyqtSignal,Qt 
from PyQt5.QtGui import QImage
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import os
from transformers import DetrForObjectDetection

from transformers import DetrForObjectDetection, DetrImageProcessor, DetrFeatureExtractor
import torch.nn as nn
import albumentations

def convert_to_lowercase(text):
    lowercase_text = ""
    for char in text:
        if char == 'G':
            lowercase_text += 'g'
        else:
            lowercase_text += char.lower()
    return lowercase_text

def label_text2int(t):
    low = convert_to_lowercase(t)
    if( low == 'eosinophil' or low == 'eosino' ): return 1 # tiểu cầu
    elif( low == 'lymphocyte' or low == 'lympho'): return 2
    elif( low == 'monocyte' or low == 'mono' ): return 3
    elif( low == 'neutrophil' or low == 'neutro' ): return 4
    elif( low == 'basophil' or low == 'baso' ): return 5
    elif( low == 'rbc' ): return 6
    # elif( low == 'platelet' or low == 'platelets' ): return 6
    else: return 0
    return 0

def label_int2text(i):
    if( i == 1 ): return 'EOSINOPHIL'
    elif( i == 2 ): return 'LYMPHOCYTE'
    elif( i == 3 ): return 'MONOCYTE'
    elif( i == 4 ): return 'NEUTROPHIL'
    elif( i == 5 ): return 'BASOPHIL'
    elif( i == 6 ): return 'RBC'
    # elif( i == 6 ): return 'PLATELET'
    else: return 'NAN'
    return 'NAN'

classes = ['basophil','eosinophil' ,'lymphocyte', 'monocyte', 'neutrophil','rbc']
num_class = 7
num_queries = 20
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
        self.in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries
    def forward(self, images):
        return self.model(images)

# Lấy đường dẫn tuyệt đối đến thư mục chứa file test.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Xây dựng đường dẫn tuyệt đối đến file detr_best.pth
detr_best_path = os.path.join(current_dir, '..', 'detr_best.pth')

model = DETRModel(num_classes=num_class, num_queries=num_queries)

state_dict = torch.load(detr_best_path, map_location=torch.device('cpu'))

model.load_state_dict(state_dict)


class BloodDetr(QThread):
    def __init__(self,imgPath):
        super(BloodDetr, self).__init__()
        self.imgPath = imgPath

    changePixmap = pyqtSignal(QImage)
    classesCountSignal = pyqtSignal(dict)
        
    def run(self):
        classes_count = {cls: 0 for cls in classes}
     
        model.eval()

        img = cv2.imread(self.imgPath)
        # img = cv2.resize(img, (600, 600))
        height, width, channel = img.shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image to an array
        image = img_to_array(image)
        resized_image_array = image.astype(np.float32) / 255.0
        nor = (resized_image_array - imagenet_mean) / imagenet_std

        X = []
        X.append(image)
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.permute(0, 3, 1, 2)

        results = model(X)
        # print("Results DeTr: \n", results)
        pred_probs = torch.softmax(results['pred_logits'], dim=-1)
        clas = torch.argmax(pred_probs, dim=-1)
        scor = torch.max(pred_probs, dim=-1).values

        box = results['pred_boxes']
        box = torch.tensor(data = box,requires_grad=False)
        box = box[0].numpy()
        print("Box1: ",box)
        box = [np.array(bbox) for bbox in albumentations.core.bbox_utils.denormalize_bboxes(box, height, width)]


        result = {'boxes': box,
                'labels': clas.detach(),
                'Scores': scor.detach()
        }
       
        boxes = result['boxes']
        clas = result['labels']
        scor = result['Scores']
        # print("Boxes: ", boxes)
        # print("Class: \n", clas)
        # print("Score: \n",scor)
        for i in range(len(boxes)):
            print("Score: ",scor[0][i])
            if scor[0][i] > 0.8:
                x = int((boxes[i][0]))
                print("x:",x)
                y = int((boxes[i][1]))
                print("y: ",y)
                x2 = int((boxes[i][2]))
                y2 = int((boxes[i][3]))
                label1 = label_int2text(clas[0][i])
                # label2 = str(round(scor[0][i].item(), 2))
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label1, (x+3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                print("Object: ", label1," with ", scor[0][i]," confident score")
                # classes_count[label1] += 1

        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytesPerLine = channel * width
        convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(600, 600, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        # self.classesCountSignal.emit(classes_count)
        
