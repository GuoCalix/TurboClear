import torch


def print_vram_state(msg, logger=None):
    """Log current device VRAM usage and return allocated, reserved, and peak GB."""
    if not torch.cuda.is_available():
        if logger:
            logger.info(f"[GPU memory]: {msg}, CUDA not available")
        return 0.0, 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if logger:
        logger.info(
            f"[GPU memory]: {msg}, allocated = {allocated:.2f} GB, "
            f"reserved = {reserved:.2f} GB, peak = {peak:.2f} GB"
        )
    return allocated, reserved, peak
