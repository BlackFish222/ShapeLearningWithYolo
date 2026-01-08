import os
import torchvision.utils as vutils
from DataSetCreation.make_dataset import make_dataset

mshape = "rectangles" #Options: rectangles or LvT (L's and T's)
pattern = "color" #Options: color or texture
size = "small" #Options: large or small
variant = "shapeonly" #Options: shapeonly, patternonly, conflict, random
mod_data_set_size = 30

def load_full_dataset(shape= mshape, pattern= pattern, size= size, variant= variant,
    batchsize=8, num_workers=2, stage="fit"):
    dm = make_dataset(shape, pattern, size, variant, batchsize, num_workers)
    dm.setup(stage)
    return dm.train_dataloader()


def save_all_images(dataloader, output_dir="TestingDataSet", pattern1_only=False, pattern1_label=0,
    max_images=mod_data_set_size):
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

            index += 1

    print(f"Saved {index} images to YOLO classification format in {output_dir}")

if __name__ == "__main__":
    dataloader = load_full_dataset(
        shape=mshape,
        pattern=pattern,
        size=size,
        variant= variant,
        batchsize=8
    )

    save_all_images(
        dataloader,
        output_dir="TestingDataSet",
        pattern1_only=False,
        max_images=mod_data_set_size,
    )
