import os
import torch
import numpy as np
import random
from PIL import Image, ImageDraw
from torchvision import transforms

from model import Generator

device = "cpu"

image_size = 256
model_path = "checkpoints/generator_best.pth"
input_image_path = "input/test.jpg"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

image = Image.open(input_image_path).convert("RGB")
image = image.resize((image_size, image_size))

mask_w = random.randint(48, 96)
mask_h = random.randint(48, 96)
x = random.randint(0, image_size - mask_w)
y = random.randint(0, image_size - mask_h)

# mask_w = 24
# mask_h = 56
# x = 30
# y = 80

mask = Image.new("L", (image_size, image_size), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([x, y, x + mask_w, y + mask_h], fill=255)

print(f"Mask placed at x={x}, y={y}, size={mask_w}x{mask_h}")

image.save(os.path.join(output_dir, "original.png"))
mask.save(os.path.join(output_dir, "mask.png"))

transform_image = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform_mask = transforms.ToTensor()

image_tensor = transform_image(image).unsqueeze(0)
mask_tensor = transform_mask(mask).unsqueeze(0)

masked_tensor = image_tensor * (1 - mask_tensor)

masked_image_np = masked_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
masked_image_np = (masked_image_np + 1) / 2
masked_image_np = np.clip(masked_image_np, 0, 1)
masked_image_np = (masked_image_np * 255).astype(np.uint8)
Image.fromarray(masked_image_np).save(os.path.join(output_dir, "masked.png"))

with torch.no_grad():
    output_tensor = generator(masked_tensor.to(device), mask_tensor.to(device))

output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
output_np = (output_np + 1) / 2
output_np = np.clip(output_np, 0, 1)
output_np = (output_np * 255).astype(np.uint8)

Image.fromarray(output_np).save(os.path.join(output_dir, "reconstructed.png"))

print("Saved:")
print(os.path.join(output_dir, "original.png"))
print(os.path.join(output_dir, "mask.png"))
print(os.path.join(output_dir, "masked.png"))
print(os.path.join(output_dir, "reconstructed.png"))
