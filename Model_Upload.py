from roboflow import Roboflow

rf = Roboflow(api_key="")
workspace = rf.workspace("")

workspace.deploy_model(
  model_type="yolov8",
  model_path="runs/classify/train/weights/best.pt",
  project_ids=[""],
  model_name=""
)

def upload_model_to_roboflow(api_key, workspace_name, model_path, project_ids, model_name):
  rf = Roboflow(api_key=api_key)
  workspace = rf.workspace(workspace_name)

  workspace.deploy_model(
    model_type="yolov8",
    model_path=model_path,
    project_ids=project_ids,
    model_name=model_name
  )