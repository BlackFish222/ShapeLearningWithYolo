from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from BaseModelTraining import 
import matplotlib.pyplot as plt
from PIL import Image
import os

def test_model(model_path: str, test_dir: str):
    model = YOLO(model_path)

    # Folder class order (your ground-truth label order)
    class_names = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith(".")
    ])

    # Model class order (what result.probs.top1 refers to)
    model_names = model.names  # dict like {0:"cat", 1:"dog"} or list
    if isinstance(model_names, dict):
        model_class_names = [model_names[i] for i in sorted(model_names.keys())]
    else:
        model_class_names = list(model_names)

    # Build mapping: model index -> folder index
    folder_index_by_name = {name: i for i, name in enumerate(class_names)}
    model_to_folder = {}
    for mi, name in enumerate(model_class_names):
        if name in folder_index_by_name:
            model_to_folder[mi] = folder_index_by_name[name]
        else:
            # If names don't match exactly, you’ll want to normalize/rename
            # For now mark as invalid
            model_to_folder[mi] = None

    true_labels, pred_labels = [], []

    for true_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(class_dir, file)
                image = Image.open(image_path).convert("RGB")

                result = model(image)[0]
                pred_model_idx = int(result.probs.top1)
                pred_folder_idx = model_to_folder.get(pred_model_idx, None)

                # Skip predictions whose class name isn't in your folder set
                if pred_folder_idx is None:
                    continue

                true_labels.append(true_idx)
                pred_labels.append(pred_folder_idx)

    print(f"Total images processed (used in CM): {len(true_labels)}")
    print(f"Folder class names: {class_names}")
    print(f"Model class names:  {model_class_names}")

    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=list(range(len(class_names)))
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Remove "background" if present (either in labels or as last row/col)
    if "background" in class_names:
        bg_i = class_names.index("background")
        class_names.pop(bg_i)
        cm = cm.copy()
        cm = cm[[i for i in range(cm.shape[0]) if i != bg_i], :]
        cm = cm[:, [i for i in range(cm.shape[1]) if i != bg_i]]

    # If "background" is not in class_names but CM has extra dim (common: last index)
    elif cm.shape[0] == len(class_names) + 1:
        # assume last row/col is background
        cm = cm[:-1, :-1]

    
    disp.plot(xticks_rotation=45, cmap="Blues", values_format="d")

    plt.title("YOLOv8 Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cm
