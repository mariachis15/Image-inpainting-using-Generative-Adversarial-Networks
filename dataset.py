import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FaceInpaintingDataset(Dataset):
    def __init__(self, root_dir, image_size=256, mask_size_min=48, mask_size_max=96):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mask_size_min = mask_size_min
        self.mask_size_max = mask_size_max

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(valid_ext)
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def generate_random_mask(self):
        mask = torch.zeros((1, self.image_size, self.image_size))
    
        num_rectangles = random.randint(1, 3)
    
        for _ in range(num_rectangles):
            mask_w = random.randint(self.mask_size_min, self.mask_size_max)
            mask_h = random.randint(self.mask_size_min, self.mask_size_max)
    
            x = random.randint(0, self.image_size - mask_w)
            y = random.randint(0, self.image_size - mask_h)
    
            mask[:, y:y + mask_h, x:x + mask_w] = 1.0
    
        return torch.clamp(mask, 0.0, 1.0)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        mask = self.generate_random_mask()
        masked_image = image * (1 - mask)

        return {
            "image": image,
            "masked_image": masked_image,
            "mask": mask,
            "image_path": image_path,
        }