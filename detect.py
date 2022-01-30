import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import train
class detect(train.trainModel):
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
