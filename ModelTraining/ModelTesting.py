from ultralytics import YOLO

model = YOLO("ModelTraining/yolov8s-cls.pt")

results = model.predict(source="ModelTraining/dataset_per_image_metadata")

