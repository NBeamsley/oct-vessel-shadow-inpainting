import numpy as np
import tensorflow as tf
from data_augmentation_3d import augment_patch_3d


class DataGenerator3D(tf.keras.utils.Sequence):
    """
    3D patch data generator with angled vessel-shadow masking.

    Takes a list of volumes and a pre-built index of valid (shadow-free) patch
    positions. Patches are sliced from the volumes on demand.

    Augmentation is controlled by n_augmented_copies:
        0 — no augmentation, original data only (use for val generator)
        N — N augmented copy per raw patch = N+1x effective dataset

    The generator virtually expands the index: the first len(valid_indices)
    batches serve raw patches, the next len(valid_indices) serve augmented
    copies, etc. Augmentation is applied before masking so the synthetic
    shadow is applied to the augmented version consistently.

    Each __getitem__ call:
      1. Determines whether this batch draw is a raw or augmented pass.
      2. Slices the corresponding (D, H, W, 1) patch from the volume.
      3. Optionally augments (contrast, flip, speckle).
      4. Generates a synthetic angled vessel-shadow mask.
      5. Returns (masked_batch, target_batch, mask_3d_batch) all (B, D, H, W, 1).
    """

    def __init__(
        self,
        volumes,
        valid_indices,
        batch_size=8,
        outshape=(8, 256, 64, 1),
        min_shadow_width=15,
        max_shadow_width=25,
        max_angle_deg=30,
        n_augmented_copies=0,
        shuffle=True,
    ):
        """
        Args:
            volumes:              list of dicts {'volume': (N,H,W,1), 'mask': (N,W)}
                                  OR list of (N,H,W,1) arrays
                                  OR list of (filename, volume) tuples
            valid_indices:        list of either:
                                  - (vol_idx, d_start, w_start) tuples
                                  - (vol_idx, d_start, w_start, axis)
            batch_size:           number of 3D chunks per batch
            outshape:             (D, H, W, 1)
            min_shadow_width:     minimum synthetic shadow width in columns
            max_shadow_width:     maximum synthetic shadow width in columns
            max_angle_deg:        maximum shadow tilt from vertical (degrees)
            n_augmented_copies:   number of augmented passes over the index.
        """
        self.volumes             = volumes
        self.valid_indices       = valid_indices
        self.batch_size          = batch_size
        self.D, self.H, self.W   = outshape[0], outshape[1], outshape[2]
        self.min_shadow_width    = min_shadow_width
        self.max_shadow_width    = max_shadow_width
        self.max_angle_deg       = max_angle_deg
        self.n_augmented_copies  = n_augmented_copies
        self.shuffle             = shuffle

        # Virtual index: n_raw + n_aug copies
        # Each entry is (original_index_position, is_augmented)
        n = len(valid_indices)
        self.nominal_size = n

        raw_entries = [(i, False) for i in range(n)]
        aug_entries = [(i, True)  for i in range(n)] * n_augmented_copies
        self.virtual_indices = raw_entries + aug_entries

        self.indices = np.arange(len(self.virtual_indices))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.virtual_indices) / self.batch_size)))

    def _get_volume(self, vol_idx):
        v = self.volumes[vol_idx]
        if isinstance(v, dict):
            return v['volume']
        if isinstance(v, (tuple, list)) and len(v) > 0 and hasattr(v[-1], 'shape'):
            return v[-1]
        return v

    # ============================================
    # patch extraction
    # ============================================

    def _extract_patch(self, entry):
        vol_idx = entry[0]
        vol = self._get_volume(vol_idx)

        if len(entry) == 3:
            _, d_start, w_start = entry
            patch = vol[
                d_start:d_start + self.D,
                :,
                w_start:w_start + self.W,
                :
            ]
            return patch.copy()

        axis = entry[3]

        if axis == "fast":
            _, d_start, w_start, _ = entry
            patch = vol[
                d_start:d_start + self.D,
                :,
                w_start:w_start + self.W,
                :
            ]
        else:  # slow axis
            _, w_start, n_start, _ = entry
            patch = vol[
                n_start:n_start + self.W,
                :,
                w_start:w_start + self.D,
                :
            ].transpose(2, 1, 0, 3)

        return patch.copy()

    # ============================================
    # Mask generation
    # ============================================

    def _make_angled_mask(self):
        """
        Build a (D, H, W, 1) binary mask for a tilted vessel shadow.

        Each call, randomise start position, angle, and thickness of 
        first and last shadow columns.  
        """
        # Independent width at start and end of the chunk
        width_start = np.random.randint(self.min_shadow_width,
                                        self.max_shadow_width + 1)
        width_end   = np.random.randint(max(self.min_shadow_width, width_start - 3),
                                        min(self.max_shadow_width, width_start + 3) + 1)

        # Random angle
        angle_deg = np.random.uniform(-self.max_angle_deg, self.max_angle_deg)
        drift     = int(round(self.D * np.tan(np.radians(angle_deg))))

        # Random start centre, keeping both the start and end edges inside the patch.
        # col_b = col_a + drift, so we need room for whichever end is wider.
        margin_a = width_start // 2 + 1
        margin_b = width_end   // 2 + 1
        # Derive valid range for col_a such that col_b = clip(col_a+drift) also fits.
        lo = max(margin_a, margin_b - drift)
        hi = min(self.W - margin_a, self.W - margin_b - drift)
        if lo >= hi:
            lo, hi = margin_a, self.W - margin_a   # fallback: ignore drift clamping
        col_a  = np.random.randint(lo, hi)
        col_b  = int(np.clip(col_a + drift, margin_b, self.W - margin_b))

        left_start  = col_a - width_start // 2
        right_start = left_start + width_start
        left_end    = col_b - width_end // 2
        right_end   = left_end + width_end

        mask       = np.zeros((self.D, self.H, self.W, 1), dtype=np.float32)
        col_starts = np.empty(self.D, dtype=int)
        col_ends   = np.empty(self.D, dtype=int)

        ts = np.linspace(0.0, 1.0, self.D)
        for d, t in enumerate(ts):
            s = int(round(left_start  + t * (left_end  - left_start)))
            e = int(round(right_start + t * (right_end - right_start)))
            s = max(0, s)
            e = min(self.W, e)
            mask[d, :, s:e, 0] = 1.0
            col_starts[d] = s
            col_ends[d]   = e

        return mask, col_starts, col_ends

    # ============================================
    # Masking
    # ============================================

    def _apply_mask(self, patch, col_starts, col_ends):
        """
        Zero the shadow columns.
        """
        masked = patch.copy()
        for d in range(self.D):
            s, e = col_starts[d], col_ends[d]
            masked[d, :, s:e, 0] = 0.0
        return masked

    # ============================================
    # Batch construction
    # ============================================

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        masked_list, target_list, mask_list = [], [], []

        for i in batch_idx:
            original_idx, is_augmented = self.virtual_indices[i]
            entry = self.valid_indices[original_idx]

            patch = self._extract_patch(entry)

            # Apply augmentation before masking
            if is_augmented:
                patch = augment_patch_3d(patch)

            mask, col_starts, col_ends = self._make_angled_mask()
            masked = self._apply_mask(patch, col_starts, col_ends)

            masked_list.append(masked)
            target_list.append(patch)
            mask_list.append(mask)

        return (
            np.asarray(masked_list, dtype=np.float32),
            np.asarray(target_list, dtype=np.float32),
            np.asarray(mask_list, dtype=np.float32),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)