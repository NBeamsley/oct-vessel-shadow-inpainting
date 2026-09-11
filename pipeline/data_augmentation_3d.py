import numpy as np

def augment_patch_3d(patch,
                     contrast_prob=0.4,
                     contrast_min_pct=2,
                     contrast_max_pct=98,
                     contrast_stretch_range=0.1,
                     hflip_prob=0.5,
                     gaussian_prob=0.3,
                     gaussian_std=0.03):
    """
    Augmentation for 3D OCT patch chunks.
    Applied to (D, H, W, 1) float32 arrays in [0, 1].

    Args:
        patch:            (D, H, W, 1) float32
        contrast_prob:    probability of contrast stretch
        contrast_min_pct: minimum percentile for contrast stretch
        contrast_max_pct: maximum percentile for contrast stretch
        hflip_prob:       probability of horizontal flip in the W plane
        gaussian_prob:    probability of Gaussian noise augmentation
        gaussian_std:     controls Gaussian noise intensity

    Returns:
        Augmented patch, same shape and dtype
    """
    patch = patch.copy()

    # === Contrast stretching ===
    # Computes the pth and qth percentile of the patch and linearly rescales so that p maps to 0 and q maps to 1.
    if np.random.random() < contrast_prob:
        perturb = np.random.uniform(-contrast_stretch_range, contrast_stretch_range)
        p_low   = np.percentile(patch, contrast_min_pct + perturb * 10)
        p_high  = np.percentile(patch, contrast_max_pct - perturb * 10)

        if p_high - p_low > 1e-4:
            patch = (patch - p_low) / (p_high - p_low)

    # === Horizontal flip in W plane ===
    if np.random.random() < hflip_prob:
        patch = patch[:, :, ::-1, :]

    # === Additive Gaussian noise ===
    if np.random.random() < gaussian_prob:
        noise = np.random.normal(0.0, gaussian_std, patch.shape).astype(np.float32)
        patch = patch + noise

    return np.clip(patch, 0.0, 1.0).astype(np.float32)