from ultralytics import YOLO


def main():
    # Memuat model YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Memulai proses training
    results = model.train(
        data="BIB_Detection-2/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        patience=15,
        name="bib_detection_v2"
    )

    print("Training selesai!")


if __name__ == "__main__":
    main()