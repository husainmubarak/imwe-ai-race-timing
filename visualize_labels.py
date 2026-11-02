
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# Lokasi dataset
IMAGE_DIR = Path("BIB_Detection-2/train/images")
LABEL_DIR = Path("BIB_Detection-2/train/labels")

# Lokasi penyimpanan hasil visualisasi
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pengaturan sampel
NUMBER_OF_IMAGES = 6
RANDOM_SEED = 42

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Mengambil seluruh gambar dari folder train
image_paths = [
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]

if not image_paths:
    raise FileNotFoundError("Tidak ada gambar yang ditemukan!")

# Memilih beberapa gambar secara acak
random.seed(RANDOM_SEED)
selected_images = random.sample(
    image_paths,
    min(NUMBER_OF_IMAGES, len(image_paths))
)

# Membuat layout visualisasi
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for index, image_path in enumerate(selected_images):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Gagal membaca gambar: {image_path.name}")
        continue

    height, width = image.shape[:2]

    # Mencari file anotasi dengan nama yang sama
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Anotasi tidak ditemukan: {label_path.name}")
        continue

    # Membaca anotasi
    with open(label_path, "r") as file:
        annotations = file.readlines()

    # Menggambar bounding box
    for annotation in annotations:

        values = annotation.strip().split()

        if len(values) != 5:
            print(f"Format anotasi tidak valid: {label_path.name}")
            continue

        class_id = int(values[0])
        center_x, center_y, box_width, box_height = map(
            float, values[1:]
        )

        # Mengubah koordinat normalisasi menjadi pixel
        x1 = int((center_x - box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        x2 = int((center_x + box_width / 2) * width)
        y2 = int((center_y + box_height / 2) * height)

        # Membatasi koordinat agar berada dalam gambar
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Menggambar kotak
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"Tag-Number ({class_id})",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # OpenCV menggunakan BGR, sedangkan Matplotlib menggunakan RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axes[index].imshow(image)
    axes[index].set_title(image_path.name)
    axes[index].axis("off")

# Menonaktifkan subplot yang tidak digunakan
for index in range(len(selected_images), len(axes)):
    axes[index].axis("off")

plt.tight_layout()

# Menyimpan hasil
output_path = OUTPUT_DIR / "bbox_preview.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nVisualisasi berhasil disimpan: {output_path.resolve()}")

plt.show()
