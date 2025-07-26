from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
import os

# Trained Model
model = YOLO("../ModelTraining/runs/classify/train2/weights/best.pt")

# Images You Would Like to Test
base_dir = "../TestingDataSet"

class_names = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
])

true_labels = []
pred_labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(base_dir, class_name)
    for file in os.listdir(class_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(class_dir, file)
            image = Image.open(image_path)

            result = model(image)[0]
            pred_index = result.probs.top1

            true_labels.append(class_idx)
            pred_labels.append(pred_index)

print(f"Total images processed: {len(true_labels)}")
print(f"Class names: {class_names}")
print(f"True label distribution: {dict((i, true_labels.count(i)) for i in set(true_labels))}")
print(f"Predicted label distribution: {dict((i, pred_labels.count(i)) for i in set(pred_labels))}")


# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')

plt.title("YOLOv8 Classification Confusion Matrix")
plt.tight_layout()
plt.show()
