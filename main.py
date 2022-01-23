from sys import path
import numpy as np
from PIL import Image 
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import transforms, datasets, models
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as patches

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
class trainModel(loadData):
    def __init__(self):
        self.device = self.getDevice()
    def createModel(self,num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def readImage(self,path):
            image = Image.open(path)
            image = image.convert("RGB")

            transform = self.getTransform()
            image = transform(image)
            print(image.size())
            return image

    def train(self,model,data_loader):
        num_epochs = 20
        model.to(self.getDevice())
    
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                        momentum=0.9, weight_decay=0.005)

        for epoch in range(num_epochs):
            model.train()
            i = 0    
            epoch_loss = 0
            for image, annotations in data_loader:
                i += 1
                image = list(image.to(self.device) for image in image)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                loss_dict = model([image[0]], [annotations[0]])
                losses = sum(loss for loss in loss_dict.values())        
                optimizer.zero_grad()
                losses.backward()
                optimizer.step() 
                print(f"epoch: {0} itter: {1}/{2} loss: {3} ".format(epoch,i,len(data_loader),losses))
                epoch_loss += losses
            print(epoch_loss)
        self.model = model
        return model
    def getDevice(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return device
    def saveModel(self,path,model=None):
        if(model==None):
            if(self.model==None):
                raise "No model have been train or loaded"
            else:
                model = self.model
            torch.save(model.state_dict(),path)
    def startTrain(self, dataset):
        model = self.createModel(3)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, collate_fn=lambda batch: tuple(zip(*batch)))
        return self.train(model,data_loader),

class detect(trainModel):
    def loadModel(self,path=None,model=None):
        if(model!=None):
            self.model = model
            return
        if(path==None):
            raise "Insert path!"
        model = self.createModel(3)
        model.load_state_dict(torch.load(path))

        model.eval()
        model.to(self.getDevice())
        self.model = model
        return model
    def detectImage(self,image,model=None):
        if(model==None):
            if(self.model==None):
                raise "No model has been loaded yet"
            else:
                model = self.model
        images = list(image.to(self.getDevice()) for image in image)
        prediction = model(images)
        return prediction

    def plot_image(self,image_tensor, annotation):
        
        fig,ax = plt.subplots(1)
        image = image_tensor.cpu().data

        ax.imshow(image.permute(1, 2, 0))
        
        for box in annotation["boxes"]:
            xmin, ymin, xmax, ymax = box.cpu().data.numpy()
            box = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(box)

        plt.show()
    def showPrediction(self,prediction,image):
        self.plot_image(image, prediction)
if __name__ == "__main__":
    # a = loadData()
    # print(torch.cuda.current_device())
    # dataset = a.loadDataset("data")
    # b= trainModel()
    # images = list(sorted(os.listdir("data/images")))
    # # b.getData(images)
    # model = b.startTrain(dataset)
    # b.saveModel("model.pt")
    c = detect()
    c.loadModel(path="model.pt")
    images = c.readImage("maskImage.png")
    print(images.size())
    prediction = c.detectImage([images])
    print(type(prediction[0]))
    c.showPrediction(prediction[0],images)
    # preds = model(images)
    # preds
    # images = list(sorted(os.listdir("data/images")))
    # model.eval()
    # preds = model(images)
    # preds
    # a.getDetail("data/annotations/maksssksksss0.xml")
    