import torch

print("=" * 50)
print("GPU TEST")
print("=" * 50)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    vram = torch.cuda.get_device_properties(0).total_memory
    print("VRAM:", round(vram / (1024 ** 3), 2), "GB")
else:
    print("❌ CUDA GPU not detected")