import os
import cv2
import numpy as np

def load_denoised_volumes(base_dir, mask_dir, target_height=256, target_width=200,
                 test_subjects=['MB1692', 'Ex-vivo1', '0P2'], n_test=100,
                 exclude=None):
    """
    Load all denoised OCT scans and their corresponding masks, returning them as per-volume lists.

    Each element in the returned lists is one complete denoised volume returned as a dict:
        {'volume': (N, H, W, 1), 'mask': (N, W), 'filename': str}

    mask_dir: directory containing mask images named <stem>_mask.png
              where <stem> matches the .npy filename without extension.
    """
    exclude = set(exclude or [])

    train_volumes = []
    test_volumes  = []

    exvivo1_id = ['0000012', '0000017', '0000022', '0000037', '0000042', '0000047']

    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.npy')])

    for filename in all_files:
        subject_id = filename.split('_')[0]

        if subject_id in exclude or filename in exclude:
            print(f"  Skipping excluded: {filename}")
            continue

        is_test = False
        if subject_id in test_subjects:
            is_test = True
        elif subject_id == 'Ex-vivo' and 'Ex-vivo1' in test_subjects:
            if any(eid in filename for eid in exvivo1_id):
                is_test = True

        file_path = os.path.join(base_dir, filename)

        # ── Load mask ───────────────────────────────────────────────────────
        mask_stem     = filename.replace('_denoised_t50.npy', '')
        mask_filename = f"{mask_stem}_mask.png"
        mask_path     = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"  Skipping {filename}: mask not found at {mask_path}")
            continue
            
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"  Skipping {filename}: could not read mask {mask_path}")
            continue

        try:
            raw_vol = np.load(file_path)
            raw_min = raw_vol.min()
            raw_max = raw_vol.max()
            print(f"  {filename} | Shape: {raw_vol.shape} | "
                  f"Range: [{raw_min:.2f}, {raw_max:.2f}] -> "
                  f"{'TEST' if is_test else 'TRAIN'}")

            if raw_max < 1.01:
                scale_factor = 1.0
            elif raw_max <= 255.0:
                scale_factor = 255.0
            else:
                scale_factor = 65535.0

            slices = []
            for i in range(raw_vol.shape[0]):
                s = cv2.resize(raw_vol[i].astype(np.float32),
                               (target_width, target_height))
                s = np.clip(s, 0, scale_factor) / scale_factor
                slices.append(s)

            vol_arr = np.expand_dims(np.array(slices, dtype=np.float32), axis=-1)
            # vol_arr: (N_bscans, H, W, 1)

            # ── Resize mask to (N_bscans, target_width) ──────────────────────
            # cv2.resize takes (W, H) — here W=target_width, H=N_bscans
            mask_resized = cv2.resize(
                mask_img,
                (target_width, raw_vol.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_arr = (mask_resized > 127).astype(np.float32)
            # mask_arr: (N_bscans, target_width)
            
            print(f"    mask shape: {mask_arr.shape}  "
                  f"shadow coverage: {mask_arr.mean()*100:.1f}%")

            # ── Bundle result ─────────────────────────────────────────────────
            entry = {'volume': vol_arr, 'mask': mask_arr, 'filename': filename}

            if is_test:
                test_volumes.append(entry)
            else:
                train_volumes.append(entry)

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    n = len(test_volumes)
    if n > 0:
        if n_test >= n:
            split        = max(1, n * 4 // 5)
            val_volumes  = test_volumes[:split]
            test_arr_out = test_volumes[split:]
        else:
            test_idx     = np.linspace(0, n - 1, n_test, dtype=int)
            val_mask     = np.ones(n, dtype=bool)
            val_mask[test_idx] = False
            val_volumes  = [test_volumes[i] for i in range(n) if val_mask[i]]
            test_arr_out = [test_volumes[i] for i in test_idx]
    else:
        val_volumes  = []
        test_arr_out = []

    print(f"\n=== Data Summary (load_denoised_volumes) ===")
    print(f"Train volumes: {len(train_volumes)}  |  "
          f"Val volumes: {len(val_volumes)}  |  Test volumes: {len(test_arr_out)}")

    return {
        'train': train_volumes,
        'val':   val_volumes,
        'test':  test_arr_out,
    }

def load_all_denoised_scans(base_dir, mask_dir, target_height=256, target_width=200,
                 test_subjects=['MB1692', 'Ex-vivo1', '0P2'], n_test=100,
                 exclude=None):
    """
    Loads all denoised OCT scans and flattens them into 2D arrays of shape
    (Total_Bscans, H, W, 1). Also returns the flattened masks of shape (Total_Bscans, W).
    """
    data = load_denoised_volumes(base_dir, mask_dir=mask_dir,
                                 target_height=target_height,
                                 target_width=target_width,
                                 test_subjects=test_subjects,
                                 n_test=0, exclude=exclude)

    train_vols = data['train']

    # 1. Flatten Training Data
    if train_vols:
        train_scans = np.concatenate([v['volume'] for v in train_vols], axis=0)
        train_masks = np.concatenate([v['mask'] for v in train_vols], axis=0)
    else:
        train_scans = np.empty((0, target_height, target_width, 1), dtype=np.float32)
        train_masks = np.empty((0, target_width), dtype=np.float32)

    # 2. Flatten Holdout Data
    holdout_vols = data['val']
    if holdout_vols:
        all_holdout_scans = np.concatenate([v['volume'] for v in holdout_vols], axis=0)
        all_holdout_masks = np.concatenate([v['mask'] for v in holdout_vols], axis=0)
    else:
        all_holdout_scans = np.empty((0, target_height, target_width, 1), dtype=np.float32)
        all_holdout_masks = np.empty((0, target_width), dtype=np.float32)

    # 3. Split Holdout into Val and Test
    n = len(all_holdout_scans)
    if n > 0:
        actual_n_test = min(n_test, max(1, n - 1))
        test_idx  = np.linspace(0, n - 1, actual_n_test, dtype=int)
        val_mask  = np.ones(n, dtype=bool)
        val_mask[test_idx] = False
        
        val_scans  = all_holdout_scans[val_mask]
        test_scans = all_holdout_scans[test_idx]
        
        val_masks  = all_holdout_masks[val_mask]
        test_masks = all_holdout_masks[test_idx]
    else:
        val_scans  = np.empty((0, target_height, target_width, 1), dtype=np.float32)
        test_scans = np.empty((0, target_height, target_width, 1), dtype=np.float32)
        test_idx   = np.array([], dtype=int)
        
        val_masks  = np.empty((0, target_width), dtype=np.float32)
        test_masks = np.empty((0, target_width), dtype=np.float32)

    print(f"\n=== Data Summary (Flattened 2D Scans) ===")
    print(f"Train: {train_scans.shape[0]}  |  "
          f"Val: {val_scans.shape[0]}  |  Test: {test_scans.shape[0]}")

    # 4. Return Output
    return {
        'train':        {'volume': train_scans, 'mask': train_masks},
        'val':          {'volume': val_scans,   'mask': val_masks},
        'test':         {'volume': test_scans,  'mask': test_masks},
        'test_indices': test_idx,
    }