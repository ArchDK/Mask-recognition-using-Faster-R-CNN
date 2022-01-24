from PIL import Image 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import data
class trainModel(data.loadData):
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
        epochs = 20
        model.to(self.getDevice())
    
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                        momentum=0.9, weight_decay=0.005)

        for epoch in range(epochs):
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
        return self.train(model,data_loader)