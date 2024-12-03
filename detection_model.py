from ultralytics import YOLO

class Detection(object):
    def __init__(self, weights):
        self.model = YOLO(weights)

    def train(self, data, epochs=10, batch=16):
        print(data, epochs, batch)
        self.model.train(data=data, epochs=epochs, batch=batch)

    def detect(self, image, threshold=0.5):
        return self.model(image, threshold)
    
