from ultralytics import YOLO

model = YOLO("ModelTraining/yolov8s-cls.pt")

#Train
#model.train(data='ShapeLearning.v2i.folder', epochs = 20, imgsz=600)

#model.val(data='ShapeLearning.v2i.folder')

#results = model.predict(source='ShapeLearning.v2i.folder')
#print(results)


#results = model.predict(source="path/to/new_images/", save=True)
results = model.predict(source="ModelTraining/dataset_per_image_metadata", save=True)