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

def train_and_predict(model_path="yolov8s-cls.pt", data_path='ShapeLearning.v2i.folder',
                      epochs=20, imgsz=600, test_data_path=None):
    model = YOLO(model_path)

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz
    )

    if test_data_path:
        results = model.predict(
            source=test_data_path,
            save=True
        )
        return results

    return model