from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

#Train
model.train(data='[insert data location here]', epochs = 20, imgs=600)

model.val(data='[insert data location here]')

results = model.predict(source='[insert data location here]')
print(results)