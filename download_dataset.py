import os
from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("phongsakhon-tongcham").project("bib_detection")
version = project.version(2)
dataset = version.download("yolov8")

print("Dataset terdonwload")