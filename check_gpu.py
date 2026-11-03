import torch

print("Versi PyTorch:", torch.__version__)
print("Versi CUDA:", torch.version.cuda)
print("CUDA tersedia:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nama GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(3, 3).cuda()
    print("\nTensor pada GPU:")
    print(x)
else:
    print("GPU belum dapat digunakan.")