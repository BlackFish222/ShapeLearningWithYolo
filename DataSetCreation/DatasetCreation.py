import os
import random
import torchvision.utils as vutils
from __init__ import make_dataset


def load_full_dataset(shape="rectangles", pattern="color", size="small", variant="coloronly",
                      batchsize=8, num_workers=2, stage="fit"):
    dm = make_dataset(shape, pattern, size, variant, batchsize, num_workers)
    dm.setup(stage)
    return dm.train_dataloader()


def split_and_save_classification_dataset(dataloader, output_dir="saved_dataset",
                                          pattern1_only=False, pattern1_label=0,
                                          extra_metadata=None, max_images=600,
                                          split_ratios=(0.7, 0.2, 0.1)):
    image_list = []

    index = 0
    for images, labels in dataloader:
        if pattern1_only:
            mask = labels == pattern1_label
            images = images[mask]
            labels = labels[mask]

        for img, label in zip(images, labels):
            if index >= max_images:
                break
            class_label = int(label.item())
            filename = f"img_{index:05d}.png"
            image_list.append((img, class_label, filename))
            index += 1

        if index >= max_images:
            break

    print(f"[INFO] Collected {len(image_list)} images, now splitting into train/val/test...")

    # Shuffle and split
    random.shuffle(image_list)
    n_total = len(image_list)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    splits = {
        "train": image_list[:n_train],
        "val": image_list[n_train:n_train + n_val],
        "test": image_list[n_train + n_val:]
    }

    # Create folders
    for split in splits:
        for _, class_label, _ in splits[split]:
            class_dir = os.path.join(output_dir, split, f"class_{class_label}")
            os.makedirs(class_dir, exist_ok=True)

    # Save images and metadata
    for split, items in splits.items():
        for img, class_label, filename in items:
            class_dir = os.path.join(output_dir, split, f"class_{class_label}")
            save_path = os.path.join(class_dir, filename)
            vutils.save_image(img, save_path, normalize=True)

    print(f"[INFO] Saved dataset to: {output_dir}")
    create_data_yaml(output_dir)


def create_data_yaml(output_dir):
    split_dir = os.path.abspath(output_dir)
    classes = sorted([d for d in os.listdir(os.path.join(output_dir, "train")) if os.path.isdir(os.path.join(output_dir, "train", d))])
    class_names = [cls.replace("class_", "") for cls in classes]

    data_yaml = {
        "train": os.path.join(split_dir, "train"),
        "val": os.path.join(split_dir, "val"),
        "test": os.path.join(split_dir, "test"),
        "nc": len(classes),
        "names": class_names
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        import yaml
        yaml.dump(data_yaml, f)

    print(f"[INFO] Wrote YOLOv8 data.yaml to: {yaml_path}")


if __name__ == "__main__":
    dataloader = load_full_dataset(
        shape="rectangles",
        pattern="color",
        size="small",
        variant="coloronly",
        batchsize=8
    )

    split_and_save_classification_dataset(
        dataloader,
        output_dir="../ModelTraining/dataset_inference_split",
        pattern1_only=False,
        max_images=600,
        extra_metadata={
            "shape": "rectangles",
            "pattern": "color",
            "variant": "coloronly"
        }
    )
