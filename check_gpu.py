import torch

print("Versi PyTorch:", torch.__version__)

if torch.cuda.is_available():

    print("CUDA tersedia!")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Jumlah GPU:", torch.cuda.device_count())

else:

    print("CUDA tidak tersedia.")
    print("Training akan menggunakan CPU jika tidak ada akselerator lain.")
