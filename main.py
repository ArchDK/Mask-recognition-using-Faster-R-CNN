import detect

import os
if __name__ == "__main__":
    datas = detect.detect()
    dataset = datas.loadDataset("data")
    images = list(sorted(os.listdir("data/images")))
    model = datas.startTrain(dataset)
    datas.saveModel("model.pt")

    datas.loadModel(path="model.pt")
    images = datas.readImage("maskImage.png")
    print(images.size())
    prediction = datas.detectImage([images])
    print(type(prediction[0]))
    datas.showPrediction(prediction[0],images)