import cv2
import csv
import easyocr
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. KONFIGURASI
# ==========================================

MODEL_PATH = "runs/detect/bib_detection_v2/weights/best.pt"

TEST_DIR = Path("BIB_Detection-2/test/images")

OUTPUT_DIR = Path("runs/ocr_test")
CROP_DIR = OUTPUT_DIR / "crops"

MAX_IMAGES = 20

IMAGE_SIZE = 640
DETECTION_CONF = 0.25


# ==========================================
# 2. MEMUAT MODEL
# ==========================================

print("Memuat model YOLOv8...")

model = YOLO(MODEL_PATH)

print("Memuat model EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=True
)

print("Kedua model berhasil dimuat!")


# ==========================================
# 3. MENYIAPKAN FOLDER
# ==========================================

CROP_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted([
    path for path in TEST_DIR.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

image_paths = image_paths[:MAX_IMAGES]

print(f"\nJumlah gambar yang akan diproses: {len(image_paths)}")


# ==========================================
# 4. PROSES DETEKSI DAN OCR
# ==========================================

results = model.predict(
    source=[str(path) for path in image_paths],
    imgsz=IMAGE_SIZE,
    conf=DETECTION_CONF,
    device=0,
    stream=True,
    verbose=False
)


csv_rows = []

for image_number, result in enumerate(results, start=1):

    image_path = Path(result.path)
    original_image = result.orig_img

    image_height, image_width = original_image.shape[:2]

    print(f"\nMemproses gambar {image_number}: {image_path.name}")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("  Tidak ditemukan BIB.")

        csv_rows.append({
            "image": image_path.name,
            "bib_index": "-",
            "detection_confidence": "-",
            "ocr_result": "",
            "ocr_confidence": "-",
            "status": "NO_BIB_DETECTED",
            "crop_path": ""
        })

        continue

    for bib_index, (box, detection_conf) in enumerate(
        zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist()
        ),
        start=1
    ):

        # ----------------------------------
        # 5. MENGAMBIL KOORDINAT BIB
        # ----------------------------------

        x1, y1, x2, y2 = map(int, box)

        # Memberikan sedikit margin pada crop
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)

        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)

        # Memotong area BIB
        crop = original_image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # ----------------------------------
        # 6. PREPROCESSING GAMBAR
        # ----------------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------
        # 7. MENYIMPAN GAMBAR CROP
        # ----------------------------------

        crop_filename = (
            f"{image_path.stem}_bib_{bib_index:02d}.jpg"
        )

        crop_path = CROP_DIR / crop_filename

        cv2.imwrite(str(crop_path), enlarged)

        # ----------------------------------
        # 8. MEMBACA ANGKA DENGAN EASYOCR
        # ----------------------------------

        ocr_results = reader.readtext(
            enlarged,
            allowlist="0123456789",
            detail=1,
            paragraph=False
        )

        # Menggabungkan hasil pembacaan
        # berdasarkan posisi dari kiri ke kanan
        ocr_results = sorted(
            ocr_results,
            key=lambda item: min(point[0] for point in item[0])
        )

        ocr_text = "".join(
            item[1] for item in ocr_results
        )

        if ocr_results:
            ocr_confidence = sum(
                item[2] for item in ocr_results
            ) / len(ocr_results)
        else:
            ocr_confidence = 0.0

        # ----------------------------------
        # 9. MENYIMPAN HASIL
        # ----------------------------------

        status = "READ" if ocr_text else "NOT_READ"

        csv_rows.append({
            "image": image_path.name,
            "bib_index": bib_index,
            "detection_confidence": round(
                detection_conf, 4
            ),
            "ocr_result": ocr_text,
            "ocr_confidence": round(
                ocr_confidence, 4
            ),
            "status": status,
            "crop_path": str(crop_path)
        })

        print(
            f"  BIB {bib_index}: {ocr_text or 'Tidak terbaca'} "
            f"(OCR confidence: {ocr_confidence:.2f})"
        )


# ==========================================
# 10. MENYIMPAN HASIL KE CSV
# ==========================================

csv_path = OUTPUT_DIR / "ocr_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "image",
        "bib_index",
        "detection_confidence",
        "ocr_result",
        "ocr_confidence",
        "status",
        "crop_path"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(csv_rows)


print("\n================================")
print("PROSES YOLOv8 + EASYOCR SELESAI")
print("================================")

print("Total gambar:", len(image_paths))
print("Total BIB terdeteksi:", sum(
    1 for row in csv_rows
    if row["status"] in ["READ", "NOT_READ"]
))
print("Total terbaca:", sum(
    1 for row in csv_rows
    if row["status"] == "READ"
))
print("Total tidak terbaca:", sum(
    1 for row in csv_rows
    if row["status"] == "NOT_READ"
))

print("\nHasil CSV:", csv_path)
print("Hasil crop:", CROP_DIR)