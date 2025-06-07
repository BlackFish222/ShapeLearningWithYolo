from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

#Train
model.train(data='C:/shapelearningtheory/ShapeLearning.v2i.folder', epochs = 20, imgs=600)

model.val(data='C:/shapelearningtheory/ShapeLearning.v2i.folder')

results = model.predict(source='C:/shapelearningtheory/ShapeLearning.v2i.folder')
print(results)