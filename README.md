# ShapeLearningWithYolo
This is the classification section of {Paper Name}.

To use this correctly:
1. Load the conda enviroment.
2. Create a base dataset using BaseDataSetCreation, there will be modification instructions in the file.
3. Create a modified data set using ModifiedDataSet creation, there will be modification instructions in the file.
4. Use ModelTesting to train yolo on your base dataset and run inferance on the new dataset.
5. You will be left with a confusion matrix showing the results.
6. If you would like to add this trained model to a roboflow project simply fill in the necessary fields in Model Upload and run it
7. If you would like to reset and run a different test simply delete the BaseDataset, ModifiedDataset, and runs directories and start back at 1.

(Optional): if you would like to swap to a different version of yolo(such as a different pretrained version) simply
swap the yolov8cls file with the new file you would like to use.

If you have questions about anything or would like to read the paper but have no way to access it please email me @
liamw5264@gmail.com