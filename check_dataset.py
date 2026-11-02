from pathlib import Path
from ultralytics.data.utils import check_det_dataset

DATA_YAML = Path("BIB_Detection-2/data.yaml")

data = check_det_dataset(str(DATA_YAML.resolve()))

print("Jumlah kelas:", data["nc"])
print("Nama kelas:", data["names"])

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

for split in ["train", "val", "test"]:
    split_path = data.get(split)

    if not split_path:
        continue

    paths = split_path if isinstance(split_path, list) else [split_path]

    for path in paths:
        image_dir = Path(path)
        images = [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        labels_dir = image_dir.parent / "labels"
        labels_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        print("Lokasi gambar:", image_dir)
        print("Jumllah gambar:", len(images))
        print("Jumlah file anotasi:", len(labels_files))
        print("Folder anotasi tersedia:", labels_dir.exists())