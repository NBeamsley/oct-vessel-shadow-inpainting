import numpy as np


def build_valid_patch_index(volumes, patch_width=64, depth=8,
                             depth_stride=4, width_stride=4,
                             max_shadow_fraction=0.05,
                             max_per_volume=None,
                             include_slow_axis=True):
    """
    Build valid patch positions using the 2D vessel shadow mask.

    volumes: list of dicts {'volume': (N, H, W, 1), 'mask': (N, W)}
        mask[bscan, col] = 1 if that column in that B-scan is shadowed
    """
    valid      = []
    n_rejected = 0

    for vol_idx, vol_data in enumerate(volumes):
        vol  = vol_data['volume']   # (N, H, W, 1)
        mask = vol_data['mask']     # (N, W) — 1 = shadowed column in that B-scan
        N, H, W, C = vol.shape

        vol_valid = []

        # Fast axis
        if N >= depth:
            for d_start in range(0, N - depth + 1, depth_stride):
                for w_start in range(0, W - patch_width + 1, width_stride):

                    # Extract the (depth, patch_width) mask region
                    # mask shape is (N, W) so index directly
                    mask_chunk  = mask[d_start:d_start + depth,
                                    w_start:w_start + patch_width]
                    shadow_frac = float(np.mean(mask_chunk))

                    if shadow_frac  <= max_shadow_fraction:
                        vol_valid.append((vol_idx, d_start, w_start, 'fast'))
                    else:
                        n_rejected += 1

        # Slow axis
        if include_slow_axis and W >= depth and N >= patch_width:
            for w_start in range(0, W - depth + 1, depth_stride):
                for n_start in range(0, N - patch_width + 1, width_stride):
                    # crop to the B-scan range: mask[n_start:n_start+patch_width, w_start:w_start+depth]
                    mask_chunk  = mask[n_start:n_start + patch_width,
                                       w_start:w_start + depth]
                    shadow_frac = float(np.mean(mask_chunk))

                    if shadow_frac <= max_shadow_fraction:
                        vol_valid.append((vol_idx, w_start, n_start, 'slow'))
                    else:
                        n_rejected += 1


        if max_per_volume is not None and len(vol_valid) > max_per_volume:
            indices   = np.random.choice(len(vol_valid), max_per_volume, replace=False)
            vol_valid = [vol_valid[i] for i in indices]

        valid.extend(vol_valid)

    if not valid:
        raise ValueError(
            "No valid patch positions found. "
            "Check that masks exist and max_shadow_fraction is appropriate."
        )

    fast_count = sum(1 for v in valid if v[3] == 'fast')
    slow_count = sum(1 for v in valid if v[3] == 'slow')
    print(f"  Valid positions: {len(valid):,}  "
          f"(fast={fast_count:,}  slow={slow_count:,})  |  "
          f"rejected: {n_rejected:,}")
    return valid