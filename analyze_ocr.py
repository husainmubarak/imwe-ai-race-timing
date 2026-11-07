import pandas as pd

# Membaca hasil OCR
df = pd.read_csv("runs/ocr_test/ocr_results.csv")

# Menghapus data yang actual_number-nya kosong atau UNKNOWN
df["actual_number"] = df["actual_number"].astype(str).str.strip()

valid_df = df[
    (df["actual_number"] != "") &
    (df["actual_number"].str.upper() != "UNKNOWN")
].copy()

# Menghitung distribusi nomor BIB sebenarnya
distribution = valid_df["actual_number"].value_counts()

print("\n=== DISTRIBUSI NOMOR BIB ===")
print(distribution)

print("\n=== RINGKASAN DATA ===")
print(f"Total sampel             : {len(df)}")
print(f"Sampel dengan nomor valid: {len(valid_df)}")
print(f"Jumlah nomor BIB unik    : {valid_df['actual_number'].nunique()}")

# Menghitung akurasi OCR
valid_df["ocr_result"] = valid_df["ocr_result"].fillna("").astype(str).str.strip()

correct = (
    valid_df["actual_number"] == valid_df["ocr_result"]
).sum()

total = len(valid_df)

if total > 0:
    accuracy = correct / total * 100
    print(f"\nAkurasi OCR: {accuracy:.2f}%")
    print(f"Prediksi benar: {correct}")
    print(f"Prediksi salah: {total - correct}")


print("\n=== PERBANDINGAN HASIL OCR ===")

for _, row in valid_df.iterrows():
    actual = row["actual_number"]
    predicted = row["ocr_result"]

    status = "BENAR" if actual == predicted else "SALAH"

    print(
        f"Actual: {actual:<10} "
        f"Prediksi: {predicted:<15} "
        f"Status: {status}"
    )