from ultralytics.data.utils import check_det_dataset

data = check_det_dataset("BIB_Detection-2/data.yaml")

print("Dataset berhasil dibaca!")
print("Train:", data["train"])
print("Validation:", data["val"])
print("Test:", data["test"])
print("Jumlah kelas:", data["nc"])
print("Nama kelas:", data["names"])
