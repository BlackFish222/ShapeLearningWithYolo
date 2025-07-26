from ultralytics import YOLO

# Make sure this file exists and is valid
model = YOLO("yolov8s-cls.pt")

#This will need to be changed later this should become the path to the dataset your using as default
model.train(
    data='ShapeLearning.v2i.folder',  # Should be either a valid .yaml config or dataset dir
    epochs=20,
    imgsz=600
)


# Run predictions on new data (inference)
results = model.predict(
   source="DataSetCreation/TestingDataSet/TestingData",  # folder with test images
   save=True  # saves results (annotated images) to runs/predict/
)
