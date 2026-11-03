
from pathlib import Path

ROOT = Path("BIB_Detection-2")

INVALID_LABELS = [
    (
        "train",
        "Image_0512_Labels_jpg.rf.54558f0ac57d0cb465e664a17938a682.txt"
    ),
    (
        "valid",
        "Image_1354_Labels_jpg.rf.74bfcd0b6c5da9bc3d912186bd8c6820.txt"
    ),
    (
        "valid",
        "Image_1372_Labels_jpg.rf.c00c796363a43d5b6250c2cd35fdb42a.txt"
    ),
    (
        "valid",
        "Image_1383_Labels_jpg.rf.80dea4b66817e745153d31607c281898.txt"
    )
]

for split, filename in INVALID_LABELS:

    label_path = ROOT / split / "labels" / filename

    print("\n" + "=" * 55)
    print("FILE:", filename)
    print("=" * 55)

    if not label_path.exists():
        print("File tidak ditemukan!")
        continue

    lines = label_path.read_text().strip().splitlines()

    for index, line in enumerate(lines, start=1):

        values = list(map(float, line.split()))

        class_id, cx, cy, bw, bh = values

        x_min = cx - bw / 2
        x_max = cx + bw / 2
        y_min = cy - bh / 2
        y_max = cy + bh / 2

        print(f"\nAnotasi ke-{index}")
        print("Class ID:", class_id)

        print(f"X min: {x_min:.6f}")
        print(f"X max: {x_max:.6f}")
        print(f"Y min: {y_min:.6f}")
        print(f"Y max: {y_max:.6f}")

        print("Kondisi:")

        if x_min < 0:
            print("- Keluar dari sisi kiri")

        if x_max > 1:
            print("- Keluar dari sisi kanan")

        if y_min < 0:
            print("- Keluar dari sisi atas")

        if y_max > 1:
            print("- Keluar dari sisi bawah")
