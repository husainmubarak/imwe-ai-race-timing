from ultralytics import YOLO

model = YOLO("yolv8n.py")

results = model.train(
    data="BIB_Detection-2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    patience=15,
    name="bib_detection_v1"
)

print("Training selesai")