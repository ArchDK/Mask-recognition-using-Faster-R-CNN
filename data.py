
import os
from torchvision import transforms, datasets, models
from PIL import Image 
import xml.etree.ElementTree as ET
import torch
class loadData:
    def __init__(self):
        pass
    def getArea(self,bnbBox):
        pos = []
        for item in bnbBox:
            pos.append(int(item.text))
        return pos
    def getType(self,name):
        type_key = {"without_mask":0,"mask_weared_incorrect":1,"with_mask":2}
        return type_key[name.text]
    def getDetail(self,file):
        tree = ET.parse(file)
        root = tree.getroot()
        position = []
        label = []
        for object in root[4:]:
            label.append(self.getType(object[0]))
            position.append(self.getArea(object[5]))
        return [position,label]
    def getTransform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
                ])
        return transform
    def getItem(self,idx,path,transform):
        fileImage = 'maksssksksss'+ str(idx) + '.png'
        fileLabel = 'maksssksksss'+ str(idx) + '.xml'
        imagePath = os.path.join(path,"images", fileImage)
        labelPath = os.path.join(path,"annotations", fileLabel).replace("\\","/")
        image = Image.open(imagePath).convert("RGB")
        image = transform(image)
        details = self.getDetail(labelPath)
        data = {}
        data["boxes"] = torch.as_tensor(details[0], dtype=torch.float32)
        data["labels"] = torch.as_tensor(details[1], dtype=torch.int64)
        data["image_id"]  = torch.tensor(idx)

        return image,data
    def loadDataset(self,path):
        dataset = []
 
        for i in range(len(list(os.listdir(path+"/images")))):
            dataset.append(self.getItem(i,path,self.getTransform()))
        return dataset