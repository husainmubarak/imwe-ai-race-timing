from ultralytics import YOLO

# Memuat model terbaik
model = YOLO(
    "runs/detect/bib_detection_v2/weights/best.pt"
)

# Evaluasi menggunakan test dataset
metrics = model.val(
    data="BIB_Detection-2/data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    plots=True,
    name="bib_detection_test_v1"
)

print("Evaluasi test dataset selesai!")