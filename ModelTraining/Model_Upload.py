from roboflow import Roboflow

rf = Roboflow(api_key="")
workspace = rf.workspace("")

workspace.deploy_model(
  model_type="yolov8",
  model_path="",
  project_ids=[""],
  model_name=""
)