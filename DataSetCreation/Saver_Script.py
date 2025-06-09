import os
import json
import torchvision.utils as vutils
from __init__  import make_dataset


def load_full_dataset(shape="rectangles", pattern="color", size="small", variant="standard",
                      batchsize=8, num_workers=2, stage="fit"):
    dm = make_dataset(shape, pattern, size, variant, batchsize, num_workers)
    dm.setup(stage)
    return dm.train_dataloader()


def save_all_images(dataloader, output_dir="saved_dataset", pattern1_only=False, pattern1_label=0,
                    extra_metadata=None, max_images=600):
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for images, labels in dataloader:
        if pattern1_only:
            mask = labels == pattern1_label
            images = images[mask]
            labels = labels[mask]

        for img, label in zip(images, labels):
            if index >= max_images:
                print(f"Saved {index} images and JSON metadata to {output_dir}")
                return

            base_filename = f"img_{index:05d}_label{label.item()}"
            image_path = os.path.join(output_dir, base_filename + ".png")
            meta_path = os.path.join(output_dir, base_filename + ".json")

            vutils.save_image(img, image_path, normalize=True)

            metadata = {
                "filename": base_filename + ".png",
                "label": int(label.item())
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            index += 1

    print(f"Saved {index} images and JSON metadata to {output_dir}")

if __name__ == "__main__":
    dataloader = load_full_dataset(
        shape="rectangles",
        pattern="color",
        size="small",
        variant="standard",
        batchsize=8
    )

    save_all_images(
        dataloader,
        output_dir="datasets/dataset_per_image_metadata",
        pattern1_only=False,
        max_images=600,
        extra_metadata={
            "shape": "rectangles",
            "pattern": "color",
            "variant": "standard"
        }
    )
