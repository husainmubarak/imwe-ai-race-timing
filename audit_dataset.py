
from pathlib import Path
from collections import Counter
import math

# 1. Lokasi utama dataset
ROOT = Path("BIB_Detection-2")

# 2. Pembagian dataset
SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test": ROOT / "test"
}

# 3. Ekstensi gambar yang diperbolehkan
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# 4. Konfigurasi kelas
NUM_CLASSES = 1


def audit_split(split_name, split_path):

    print(f"\n{'=' * 45}")
    print(f"PEMERIKSAAN: {split_name.upper()}")
    print(f"{'=' * 45}")

    image_dir = split_path / "images"
    label_dir = split_path / "labels"

    # Memastikan folder tersedia
    if not image_dir.exists() or not label_dir.exists():
        print("ERROR: Folder images atau labels tidak ditemukan!")
        return

    # Mengambil seluruh gambar
    images = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    # Mengambil seluruh file anotasi
    labels = {
        p.stem: p
        for p in label_dir.glob("*.txt")
    }

    # Memeriksa kecocokan nama file
    missing_labels = set(images) - set(labels)
    orphan_labels = set(labels) - set(images)

    print(f"Jumlah gambar       : {len(images)}")
    print(f"Jumlah file label   : {len(labels)}")
    print(f"Gambar tanpa label  : {len(missing_labels)}")
    print(f"Label tanpa gambar  : {len(orphan_labels)}")

    valid_boxes = 0
    invalid_boxes = 0
    empty_labels = 0

    class_counts = Counter()
    box_widths = []
    box_heights = []

    # Memeriksa setiap file anotasi
    for label_path in labels.values():

        content = label_path.read_text(encoding="utf-8").strip()

        # File kosong dapat digunakan sebagai gambar tanpa objek
        if not content:
            empty_labels += 1
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1
        ):

            values = line.split()

            # Format YOLO harus memiliki 5 nilai
            if len(values) != 5:
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}, "
                    f"baris {line_number}: jumlah nilai bukan 5"
                )
                continue

            try:
                numbers = [float(value) for value in values]

            except ValueError:
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}, "
                    f"baris {line_number}: nilai bukan angka"
                )
                continue

            # Memeriksa apakah semua nilai berupa angka valid
            if not all(math.isfinite(value) for value in numbers):
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}, "
                    f"baris {line_number}: angka tidak valid"
                )
                continue

            class_value, cx, cy, bw, bh = numbers

            # Memeriksa ID kelas
            if not class_value.is_integer():
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}: ID kelas bukan integer"
                )
                continue

            class_id = int(class_value)

            if not 0 <= class_id < NUM_CLASSES:
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}: ID kelas tidak dikenal"
                )
                continue

            # Memeriksa koordinat dan ukuran bounding box
            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
            ):
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}: "
                    f"koordinat atau ukuran tidak valid"
                )
                continue

            # Menghitung batas bounding box
            x_min = cx - bw / 2
            x_max = cx + bw / 2

            y_min = cy - bh / 2
            y_max = cy + bh / 2

            # Memastikan kotak tidak keluar dari gambar
            if (
                x_min < -1e-5
                or x_max > 1 + 1e-5
                or y_min < -1e-5
                or y_max > 1 + 1e-5
            ):
                invalid_boxes += 1
                print(
                    f"[ERROR] {label_path.name}: "
                    f"bounding box keluar dari batas gambar"
                )
                continue

            # Jika semua pemeriksaan berhasil
            valid_boxes += 1
            class_counts[class_id] += 1

            box_widths.append(bw)
            box_heights.append(bh)

    # Menampilkan ringkasan
    print("\n--- HASIL ANOTASI ---")
    print(f"Bounding box valid  : {valid_boxes}")
    print(f"Bounding box invalid: {invalid_boxes}")
    print(f"File label kosong   : {empty_labels}")
    print(f"Distribusi kelas    : {dict(class_counts)}")

    if box_widths:
        print("\n--- UKURAN BOUNDING BOX ---")
        print(f"Lebar terkecil      : {min(box_widths):.4f}")
        print(f"Lebar rata-rata     : {sum(box_widths) / len(box_widths):.4f}")
        print(f"Lebar terbesar      : {max(box_widths):.4f}")

        print(f"Tinggi terkecil     : {min(box_heights):.4f}")
        print(f"Tinggi rata-rata    : {sum(box_heights) / len(box_heights):.4f}")
        print(f"Tinggi terbesar     : {max(box_heights):.4f}")

    print("\n--- STATUS ---")

    if (
        not missing_labels
        and not orphan_labels
        and invalid_boxes == 0
    ):
        print("LULUS: Struktur dan format anotasi valid.")
    else:
        print("PERLU DIPERIKSA: Ditemukan masalah pada dataset.")


# 5. Menjalankan pemeriksaan seluruh pembagian dataset
if __name__ == "__main__":

    for split_name, split_path in SPLITS.items():
        audit_split(split_name, split_path)

    print("\nPemeriksaan seluruh dataset selesai!")
