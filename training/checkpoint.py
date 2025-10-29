import os
import torch


def save_checkpoint(
    outdir,
    tag,
    generator,
    discriminator_period,
    discriminator_scale,
    ema_generator,
    generator_optimizer,
    discriminator_optimizer,
    generator_scheduler,
    discriminator_scheduler,
    scaler,
    step,
    epoch,
):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{tag}.pt")

    checkpoint = {
        "generator": generator.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "discriminator_period": discriminator_period.state_dict(),
        "discriminator_scale": discriminator_scale.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "generator_scheduler": generator_scheduler.state_dict(),
        "discriminator_scheduler": discriminator_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
    }

    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved → {path}")


def maybe_restore_checkpoint(
    path,
    generator,
    discriminator_period,
    discriminator_scale,
    ema_generator,
    generator_optimizer,
    discriminator_optimizer,
    generator_scheduler,
    discriminator_scheduler,
    scaler,
    device,
):
    checkpoint = torch.load(path, map_location=device)

    generator.load_state_dict(checkpoint["generator"])
    ema_generator.load_state_dict(checkpoint["ema_generator"])
    discriminator_period.load_state_dict(checkpoint["discriminator_period"])
    discriminator_scale.load_state_dict(checkpoint["discriminator_scale"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    if "generator_scheduler" in checkpoint:
        generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
        print("[*] Restored generator LR scheduler state.")

    if "discriminator_scheduler" in checkpoint:
        discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
        print("[*] Restored discriminator LR scheduler state.")

    print(f"[checkpoint] Restored from {path}")
    print(f"[*] Current LR after restore: {generator_scheduler.get_last_lr()[0]:.6f}")
    return checkpoint["step"], checkpoint["epoch"]
