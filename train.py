import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FaceInpaintingDataset
from model import Generator, Discriminator

device = "cpu"

image_size = 256
batch_size = 2
epochs = 1
learning_rate = 0.0002
lambda_l1 = 100

print("Loading datasets...")
train_dataset = FaceInpaintingDataset("data/train", image_size=image_size)
val_dataset = FaceInpaintingDataset("data/val", image_size=image_size)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
)

os.makedirs("checkpoints", exist_ok=True)

print("Starting training...")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")

    generator.train()
    discriminator.train()

    running_g_loss = 0.0
    running_d_loss = 0.0

    for i, batch in enumerate(train_loader):
        if i % 50 == 0:
            print(f"Batch {i}/{len(train_loader)}")

        real_images = batch["image"].to(device)
        masked_images = batch["masked_image"].to(device)
        masks = batch["mask"].to(device)

        fake_images = generator(masked_images, masks)

        # Train Discriminator
        optimizer_d.zero_grad()

        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images.detach())

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss_real = criterion_gan(real_pred, real_labels)
        d_loss_fake = criterion_gan(fake_pred, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        fake_pred = discriminator(fake_images)
        adv_loss = criterion_gan(fake_pred, real_labels)
        l1_loss = criterion_l1(fake_images, real_images)

        g_loss = adv_loss + lambda_l1 * l1_loss

        g_loss.backward()
        optimizer_g.step()

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    avg_g_loss = running_g_loss / len(train_loader)
    avg_d_loss = running_d_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] G Loss: {avg_g_loss:.4f} D Loss: {avg_d_loss:.4f}"
    )

    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch + 1}.pth")
    torch.save(
        discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch + 1}.pth"
    )

print("Training finished.")
