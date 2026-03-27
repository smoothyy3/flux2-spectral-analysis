from PIL import Image
import os

path = r"C:\Users\Jonas\Downloads\generation-7c4444ca-c2a6-4c4b-a871-72958663812e.png"  # change to your filename

# File basics
size = os.path.getsize(path)
img = Image.open(path)
print(f"Format: {img.format}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")
print(f"File size: {size / 1024:.1f} KB")

# Check all metadata
print(f"\nInfo keys: {list(img.info.keys())}")
for key, val in img.info.items():
    val_str = str(val)[:200]
    print(f"  {key}: {val_str}")

# Check for C2PA / XMP / EXIF
if hasattr(img, '_getexif') and img._getexif():
    print(f"\nEXIF entries: {len(img._getexif())}")
else:
    print("\nNo EXIF data")