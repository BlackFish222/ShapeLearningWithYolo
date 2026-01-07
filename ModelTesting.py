from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
import os

from BaseDatasetCreation import base_data_set_size


def canon(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("class_"):
        name = name[len("class_"):]
    return name


def train_model(
    model_path: str = "yolov8s-cls.pt",
    data_path: str = "BaseDataSet",
    epochs: int = 20,
    imgsz: int = base_data_set_size,
):
    print("=== TRAIN ===")
    print("cwd:", os.getcwd())
    print("model_path:", Path(model_path).resolve())
    print("data_path:", Path(data_path).resolve())

    mp = Path(model_path)
    dp = Path(data_path)

    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp.resolve()}")
    if not dp.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dp.resolve()}")

    model = YOLO(str(mp))
    train_results = model.train(data=str(dp), epochs=epochs, imgsz=imgsz)

    # Find save_dir robustly
    save_dir = getattr(train_results, "save_dir", None)
    if save_dir is None and hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Could not determine Ultralytics save_dir after training.")

    save_dir = Path(save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    if best_pt.exists():
        weights_path = best_pt
    elif last_pt.exists():
        weights_path = last_pt
    else:
        raise RuntimeError(f"No weights found in {save_dir / 'weights'}")

    print("Saved weights:", weights_path.resolve())
    return str(weights_path)


def confusion_matrix_eval(model_path: str, test_dir: str, save_fig: str = "confusion_matrix.png"):
    print("\n=== EVAL (CONFUSION MATRIX) ===")
    mp = Path(model_path)
    td = Path(test_dir)

    print("model_path:", mp.resolve())
    print("test_dir:", td.resolve())

    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp.resolve()}")
    if not td.exists():
        raise FileNotFoundError(f"Test directory not found: {td.resolve()}")

    model = YOLO(str(mp))

    # Folder-based class names (ground truth)
    class_names = sorted([d.name for d in td.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_names:
        raise RuntimeError(
            f"No class folders found in {td.resolve()}\n"
            f"Expected: {test_dir}\\classA\\img.jpg"
        )

    # Model class names/order
    model_names = model.names
    if isinstance(model_names, dict):
        model_class_names = [model_names[i] for i in sorted(model_names.keys())]
    else:
        model_class_names = list(model_names)

    print("Folder classes:", class_names)
    print("Model classes: ", model_class_names)

    # Canonical mapping so '0' matches 'class_0'
    folder_index_by_name = {canon(name): i for i, name in enumerate(class_names)}
    model_to_folder = {mi: folder_index_by_name.get(canon(name)) for mi, name in enumerate(model_class_names)}

    true_labels, pred_labels = [], []
    total_found, total_used = 0, 0

    for true_idx, class_name in enumerate(class_names):
        class_dir = td / class_name
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                total_found += 1
                image = Image.open(p).convert("RGB")

                result = model(image)[0]
                pred_model_idx = int(result.probs.top1)
                pred_folder_idx = model_to_folder.get(pred_model_idx)

                if pred_folder_idx is None:
                    # This would now only happen if model has a class name not present in folders
                    continue

                total_used += 1
                true_labels.append(true_idx)
                pred_labels.append(pred_folder_idx)

    print("Images found:", total_found)
    print("Images used :", total_used)

    if total_used == 0:
        raise RuntimeError(
            "0 images were used in the confusion matrix.\n"
            "Even after canon() mapping, no predicted classes matched folder classes.\n"
            "Check the printed Folder classes vs Model classes."
        )

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))

    # Remove background if present
    if "background" in class_names:
        bg_i = class_names.index("background")
        keep = [i for i in range(len(class_names)) if i != bg_i]
        class_names = [class_names[i] for i in keep]
        cm = cm[keep][:, keep]
    elif cm.shape[0] == len(class_names) + 1:
        cm = cm[:-1, :-1]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="Blues", values_format="d")
    plt.title("YOLOv8 Classification Confusion Matrix")
    plt.tight_layout()

    out = Path(save_fig)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved confusion matrix figure:", out.resolve())

    plt.show()
    return cm


def main():
    # Adjust if needed
    train_data = "BaseDataSet"
    test_data = "TestingDataSet"

    weights_path = train_model(
        model_path="yolov8s-cls.pt",
        data_path=train_data,
        epochs=20,
        imgsz=base_data_set_size,
    )

    confusion_matrix_eval(
        model_path=weights_path,
        test_dir=test_data,
        save_fig="confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
