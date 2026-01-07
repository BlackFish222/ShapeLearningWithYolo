import argparse
import os
import sys
from typing import Tuple, List
import torch

from BaseDatasetCreation import load_full_dataset as load_base_dataset, split_and_save_classification_dataset
from ModifiedDataSetCreator import load_full_dataset as load_test_dataset, save_all_images
from BaseModelTraining import train_and_predict
from ModelTesting import test_model
from Model_Upload import upload_model_to_roboflow


class ShapeLearningPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def create_base_dataset(
            self,
            output_dir: str = "BaseDataSet",
            shape: str = "rectangles",
            pattern: str = "color",
            size: str = "small",
            variant: str = "standard",
            batchsize: int = 8,
            max_images: int = 50,
            split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ):
        """Create and split the base dataset"""
        print(f"\n=== Creating Base Dataset ===")
        print(f"Output: {output_dir}")
        print(f"Shape: {shape}, Pattern: {pattern}, Size: {size}, Variant: {variant}")
        print(f"Max images: {max_images}, Split ratios: {split_ratios}")

        # Load dataset
        dataloader = load_base_dataset(
            shape=shape,
            pattern=pattern,
            size=size,
            variant=variant,
            batchsize=batchsize
        )

        # Split and save
        split_and_save_classification_dataset(
            dataloader,
            output_dir=output_dir,
            pattern1_only=False,
            max_images=max_images,
            split_ratios=split_ratios,
            extra_metadata={
                "shape": shape,
                "pattern": pattern,
                "variant": variant
            }
        )

        return os.path.abspath(output_dir)

    def create_testing_dataset(
            self,
            output_dir: str = "TestingDataSet",
            shape: str = "rectangles",
            pattern: str = "color",
            size: str = "small",
            variant: str = "shapeonly",
            batchsize: int = 8,
            max_images: int = 50
    ):
        """Create testing dataset"""
        print(f"\n=== Creating Testing Dataset ===")
        print(f"Output: {output_dir}")
        print(f"Shape: {shape}, Pattern: {pattern}, Size: {size}, Variant: {variant}")

        # Load dataset
        dataloader = load_test_dataset(
            shape=shape,
            pattern=pattern,
            size=size,
            variant=variant,
            batchsize=batchsize
        )

        # Save all images
        save_all_images(
            dataloader,
            output_dir=output_dir,
            pattern1_only=False,
            max_images=max_images,
            extra_metadata={
                "shape": shape,
                "pattern": pattern,
                "variant": variant
            }
        )

        return os.path.abspath(output_dir)

    def train_model(
            self,
            data_path: str = "BaseDataSet",
            model_type: str = "yolov8s-cls.pt",
            epochs: int = 20,
            imgsz: int = 10,
            test_data_path: str = None
    ):
        """Train the YOLO model"""
        print(f"\n=== Training Model ===")
        print(f"Data path: {data_path}, Model type: {model_type}")
        print(f"Epochs: {epochs}, Image size: {imgsz}")

        # Train and optionally predict
        model = train_and_predict(
            model_path=model_type,
            data_path=data_path,
            epochs=epochs,
            imgsz=imgsz,
            test_data_path=test_data_path
        )

        return model

    def test_model(
            self,
            model_path: str,
            test_dir: str
    ):
        """Test the trained model"""
        print(f"\n=== Testing Model ===")
        print(f"Model path: {model_path}")
        print(f"Test directory: {test_dir}")

        results = test_model(
            model_path=model_path,
            test_dir=test_dir
        )

        return results

    def upload_model(
            self,
            api_key: str,
            workspace_name: str,
            model_path: str,
            project_ids: List[str],
            model_name: str
    ):
        """Upload model to Roboflow"""
        print(f"\n=== Uploading Model to Roboflow ===")
        print(f"Workspace: {workspace_name}, Model: {model_name}")

        upload_model_to_roboflow(
            api_key=api_key,
            workspace_name=workspace_name,
            model_path=model_path,
            project_ids=project_ids,
            model_name=model_name
        )


def main():
    # Initialize the pipeline
    pipeline = ShapeLearningPipeline()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Central Hub for Shape Learning Project")

    # Dataset creation arguments
    parser.add_argument("--create_base_dataset", action="store_true", help="Create base dataset")
    parser.add_argument("--create_testing_dataset", action="store_true", help="Create testing dataset")
    parser.add_argument("--output_dir", default="../BaseDataset", help="Output directory for datasets")
    parser.add_argument("--shape", default="rectangles", help="Shape type for dataset")
    parser.add_argument("--pattern", default="color", help="Pattern type for dataset")
    parser.add_argument("--size", default="small", help="Size of shapes in dataset")
    parser.add_argument("--variant", default="standard", help="Dataset variant")
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--max_images", type=int, default=600, help="Maximum number of images to generate")
    parser.add_argument("--split_ratios", default="0.7,0.2,0.1", help="Train/val/test split ratios")

    # Model training arguments
    parser.add_argument("--train_model", action="store_true", help="Train the model")
    parser.add_argument("--data_path", default="ShapeLearning.v2i.folder", help="Path to training data")
    parser.add_argument("--model_type", default="yolov8s-cls.pt", help="Model type for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=600, help="Image size for training")
    parser.add_argument("--test_after_train", action="store_true", help="Run testing after training")

    # Model testing arguments
    parser.add_argument("--test_model", action="store_true", help="Test the model")
    parser.add_argument("--model_path", default="../ModelTraining/runs/classify/train/weights/best.pt",
                        help="Path to trained model")
    parser.add_argument("--test_dir", default="TestingDataSet", help="Directory with test images")

    # Model upload arguments
    parser.add_argument("--upload_model", action="store_true", help="Upload model to Roboflow")
    parser.add_argument("--api_key", help="Roboflow API key")
    parser.add_argument("--workspace_name", help="Roboflow workspace name")
    parser.add_argument("--project_ids", help="Comma-separated list of project IDs")
    parser.add_argument("--upload_model_name", help="Name for uploaded model")

    args = parser.parse_args()

    # Convert split ratios from string to tuple of floats
    split_ratios = tuple(map(float, args.split_ratios.split(",")))

    # Execute requested operations
    try:
        if args.create_base_dataset:
            pipeline.create_base_dataset(
                output_dir=args.output_dir,
                shape=args.shape,
                pattern=args.pattern,
                size=args.size,
                variant=args.variant,
                batchsize=args.batchsize,
                max_images=args.max_images,
                split_ratios=split_ratios
            )

        if args.create_testing_dataset:
            test_dir = pipeline.create_testing_dataset(
                output_dir=args.output_dir,
                shape=args.shape,
                pattern=args.pattern,
                size=args.size,
                variant=args.variant,
                batchsize=args.batchsize,
                max_images=args.max_images
            )

        if args.train_model:
            model = pipeline.train_model(
                data_path=args.data_path,
                model_type=args.model_type,
                epochs=args.epochs,
                imgsz=args.imgsz,
                test_data_path=args.test_dir if args.test_after_train else None
            )

        if args.test_model:
            pipeline.test_model(
                model_path=args.model_path,
                test_dir=args.test_dir
            )

        if args.upload_model:
            if not all([args.api_key, args.workspace_name, args.project_ids, args.upload_model_name]):
                print(
                    "Error: All upload parameters (api_key, workspace_name, project_ids, upload_model_name) are required")
                return

            pipeline.upload_model(
                api_key=args.api_key,
                workspace_name=args.workspace_name,
                model_path=args.model_path,
                project_ids=args.project_ids.split(","),
                model_name=args.upload_model_name
            )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()