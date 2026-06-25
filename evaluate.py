import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips as lpips_pkg

from model import Generator
from dataset import FaceInpaintingDataset


def denorm(x):
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


@torch.no_grad()
def evaluate(generator, loader, device, compute_fid=True):
    generator.eval()

    lpips_fn = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    fid = (
        FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        if compute_fid
        else None
    )

    ratios_list, psnr_list, ssim_list, lpips_list = [], [], [], []
    sse_hole_list, n_hole_list, sae_hole_list = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        real = batch["image"].to(device, non_blocking=True)
        masked = batch["masked_image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        fake = generator(masked, mask)
        completed = fake * mask + real * (1.0 - mask)

        real01 = denorm(real)
        comp01 = denorm(completed)
        ratios = mask.flatten(1).mean(dim=1)

        mse = ((comp01 - real01) ** 2).mean(dim=[1, 2, 3])
        psnr_full = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))

        ssim_per = ssim_fn(comp01, real01, data_range=1.0, reduction="none")
        lpips_vals = lpips_fn(completed, real).flatten()

        diff_sq = (comp01 - real01) ** 2
        mask3 = mask.expand_as(diff_sq)
        sse_hole = (diff_sq * mask3).sum(dim=[1, 2, 3])
        n_hole = mask3.sum(dim=[1, 2, 3])
        sae_hole = ((comp01 - real01).abs() * mask3).sum(dim=[1, 2, 3])

        if fid is not None:
            fid.update(real01, real=True)
            fid.update(comp01, real=False)

        ratios_list.append(ratios.cpu())
        psnr_list.append(psnr_full.cpu())
        ssim_list.append(ssim_per.cpu())
        lpips_list.append(lpips_vals.cpu())
        sse_hole_list.append(sse_hole.cpu())
        n_hole_list.append(n_hole.cpu())
        sae_hole_list.append(sae_hole.cpu())

    ratios = torch.cat(ratios_list)
    psnr_arr = torch.cat(psnr_list)
    ssim_arr = torch.cat(ssim_list)
    lpips_arr = torch.cat(lpips_list)
    sse_hole = torch.cat(sse_hole_list)
    n_hole_t = torch.cat(n_hole_list)
    sae_hole = torch.cat(sae_hole_list)

    buckets = [
        ("all", 0.0, 1.01),
        ("small (<10%)", 0.0, 0.10),
        ("medium (10-20%)", 0.10, 0.20),
        ("large (>=20%)", 0.20, 1.01),
    ]

    results = {}
    for name, lo, hi in buckets:
        sel = (ratios >= lo) & (ratios < hi)
        n = int(sel.sum().item())
        if n == 0:
            results[name] = {"n": 0}
            continue

        pooled_mse_hole = sse_hole[sel].sum() / n_hole_t[sel].sum().clamp(min=1.0)
        pooled_psnr_hole = 10.0 * torch.log10(1.0 / pooled_mse_hole.clamp(min=1e-12))
        pooled_l1_hole = sae_hole[sel].sum() / n_hole_t[sel].sum().clamp(min=1.0)

        results[name] = {
            "n": n,
            "PSNR_full": psnr_arr[sel].mean().item(),
            "SSIM_full": ssim_arr[sel].mean().item(),
            "PSNR_hole": pooled_psnr_hole.item(),
            "LPIPS": lpips_arr[sel].mean().item(),
            "L1_hole": pooled_l1_hole.item(),
        }

    if fid is not None and results.get("all", {}).get("n", 0) > 0:
        results["all"]["FID"] = fid.compute().item()

    return results, ratios


def print_results_table(results):
    has_fid = "FID" in results.get("all", {})
    cols = ["Bucket", "n", "PSNR_full", "SSIM_full", "PSNR_hole", "LPIPS", "L1_hole"]
    widths = [16, 5, 10, 9, 10, 7, 8]
    if has_fid:
        cols.append("FID")
        widths.append(7)

    def fmt(values):
        return "| " + " | ".join(f"{v:<{w}}" for v, w in zip(values, widths)) + " |"

    print()
    print(fmt(cols))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for name, m in results.items():
        if m.get("n", 0) == 0:
            continue
        row = [
            name,
            str(m["n"]),
            f"{m['PSNR_full']:.3f}",
            f"{m['SSIM_full']:.4f}",
            f"{m['PSNR_hole']:.3f}",
            f"{m['LPIPS']:.4f}",
            f"{m['L1_hole']:.4f}",
        ]
        if has_fid:
            row.append(f"{m['FID']:.3f}" if "FID" in m else "-")
        print(fmt(row))


def plot_mask_histogram(ratios, save_path):
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratios.numpy() * 100, bins=30, edgecolor="black")
    ax.axvline(10, color="red", linestyle="--", label="10%")
    ax.axvline(20, color="orange", linestyle="--", label="20%")
    ax.set_xlabel("Hole area (% of image)")
    ax.set_ylabel("Number of validation images")
    ax.set_title("Mask area distribution on validation set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def show_grid(generator, loader, device, n=6, save_path=None):
    generator.eval()
    batch = next(iter(loader))
    real = batch["image"][:n].to(device)
    masked = batch["masked_image"][:n].to(device)
    mask = batch["mask"][:n].to(device)
    fake = generator(masked, mask)
    completed = fake * mask + real * (1 - mask)

    real01 = denorm(real).cpu()
    masked01 = denorm(masked).cpu()
    comp01 = denorm(completed).cpu()
    mask_v = mask.cpu()

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    cols = ["Masked input", "Mask", "Completed", "Ground truth"]
    for i in range(n):
        axes[i, 0].imshow(masked01[i].permute(1, 2, 0))
        axes[i, 1].imshow(mask_v[i, 0], cmap="gray")
        axes[i, 2].imshow(comp01[i].permute(1, 2, 0))
        axes[i, 3].imshow(real01[i].permute(1, 2, 0))
        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(cols[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, default="./data/val")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/generator_best.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--grid_out", type=str, default="./qualitative_grid.png")
    parser.add_argument("--hist_out", type=str, default="./mask_histogram.png")
    parser.add_argument("--results_out", type=str, default="./results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = FaceInpaintingDataset(args.val_dir, image_size=args.image_size)
    print(f"Val images: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.ckpt, map_location=device))

    results, ratios = evaluate(
        generator, val_loader, device, compute_fid=not args.no_fid
    )

    print_results_table(results)

    plot_mask_histogram(ratios, args.hist_out)
    print(f"\nMask histogram saved to: {args.hist_out}")

    Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
    show_grid(generator, val_loader, device, n=6, save_path=args.grid_out)
    print(f"Grid saved to: {args.grid_out}")

    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.results_out}")


if __name__ == "__main__":
    main()
