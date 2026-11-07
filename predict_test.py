from ultralytics import YOLO

# Memuat model terbaik hasil training
model = YOLO(
    "runs/detect/bib_detection_v2/weights/best.pt"
)

# Melakukan deteksi pada gambar test
results = model.predict(
    source="BIB_Detection-2/test/images",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device=0,
    save=True,
    project="runs/detect",
    name="bib_test_predictions"
)

print("Prediksi selesai!")
print("Jumlah gambar:", len(results))
print("Hasil tersimpan di runs/detect/bib_test_predictions")