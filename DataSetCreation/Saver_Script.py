import os
import json
import torchvision.utils as vutils
from __init__  import make_dataset


def load_full_dataset(shape="rectangles", pattern="color", size="small", variant="shapeonly",
                      batchsize=8, num_workers=2, stage="fit"):
    dm = make_dataset(shape, pattern, size, variant, batchsize, num_workers)
    dm.setup(stage)
    return dm.train_dataloader()


def save_all_images(dataloader, output_dir="saved_dataset", pattern1_only=False, pattern1_label=0,
                    extra_metadata=None, max_images=600):
    index = 0
    for images, labels in dataloader:
        if pattern1_only:
            mask = labels == pattern1_label
            images = images[mask]
            labels = labels[mask]

        for img, label in zip(images, labels):
            if index >= max_images:
                print(f"Saved {index} images to YOLO classification format in {output_dir}")
                return

            label_int = int(label.item())
            class_dir = os.path.join(output_dir, str(label_int))
            os.makedirs(class_dir, exist_ok=True)

            filename = f"img_{index:05d}.png"
            image_path = os.path.join(class_dir, filename)

            vutils.save_image(img, image_path, normalize=True)

            # Optional: metadata
            # metadata = {
            #     "filename": filename,
            #     "label": label_int
            # }
            # with open(os.path.join(class_dir, f"{filename}.json"), "w") as f:
            #     json.dump(metadata, f, indent=2)

            index += 1

    print(f"Saved {index} images to YOLO classification format in {output_dir}")

if __name__ == "__main__":
    dataloader = load_full_dataset(
        shape="rectangles",
        pattern="color",
        size="small",
        variant="shapeonly",
        batchsize=8
    )

    save_all_images(
        dataloader,
        output_dir="../CollectedData/TestingData",
        pattern1_only=False,
        max_images=600,
        extra_metadata={
            "shape": "rectangles",
            "pattern": "color",
            "variant": "shapeonly",
        }
    )
