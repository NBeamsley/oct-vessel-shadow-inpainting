import os
import time
import numpy as np
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from pipeline.denoised_loader_masks import load_denoised_volumes
from pipeline.data_generator_3d import DataGenerator3D
from pipeline.patch_extractor_3d_masks import build_valid_patch_index

# ============================================
#               Variables
# ============================================
RUN_NAME              = "unet_3d_v1"

PATCH_WIDTH           = 64   # columns per patch 
DEPTH_CHUNKS          = 16   # B-scan chunk size
PATCH_SHAPE           = (DEPTH_CHUNKS, 256, PATCH_WIDTH, 1)
BATCH_SIZE            = 16  
EPOCHS                = 150
STEPS_PER_EPOCH       = 5000
CHECKPOINT_EVERY      = 25
N_AUGMENTED_COPIES    = 2

# Learning rate schedule
AE_LR_START           = 1e-4
DISC_LR_START         = 1e-4
LR_PATIENCE           = 10
LR_FACTOR             = 0.5
MIN_DELTA             = 1e-4
MIN_LR                = 1e-7
EARLY_STOP_PATIENCE   = 25
best_val_ae_initial = np.inf

# Weights 
REC_WEIGHT            = 5.0
ADV_WEIGHT            = 0.2
STYLE_WEIGHT          = 0.5
CONTENT_WEIGHT        = 0.02
UNMASKED_WEIGHT       = 0.1

print(f"weights: rec={REC_WEIGHT}, adv={ADV_WEIGHT}, style={STYLE_WEIGHT}, content={CONTENT_WEIGHT}, unmasked={UNMASKED_WEIGHT}")

# ============================================
#              Data Loading
# ============================================
print("Loading OCT volumes...")
base_dir = "data/denoised"
mask_dir = "data/masks"

data = load_denoised_volumes(
    base_dir,
    mask_dir=mask_dir,
    target_height=256,
    target_width=200,
    test_subjects=['test'],
    exclude=['exclude'],
)

train_volumes = data['train']
val_volumes   = data['val']
test_volumes  = data['test']

# Save test split for evaluation
test_data_dir = os.path.join('test_outputs', RUN_NAME)
os.makedirs(test_data_dir, exist_ok=True)
np.save(os.path.join(test_data_dir, 'test_volumes.npy'), np.asarray(test_volumes, dtype=object))
np.save(os.path.join(test_data_dir, 'test_filenames.npy'), np.asarray([entry['filename'] for entry in test_volumes], dtype=str))
print(f"Saved raw test split to {test_data_dir} ({len(test_volumes)} volumes)")

del data

gc.collect()

print(f"\nVolume split for slab training:")
print(f"Train volumes: {len(train_volumes)}")
print(f"Val volumes:   {len(val_volumes)}")
print(f"Test volumes:  {len(test_volumes)}")

print(f"\nBuilding valid batch indices ")

train_indices = build_valid_patch_index(train_volumes, patch_width=PATCH_WIDTH, depth=DEPTH_CHUNKS,
                             depth_stride=4, width_stride=4, 
                             max_shadow_fraction=0.05, max_per_volume=None,
                             include_slow_axis=False)

val_indices = build_valid_patch_index(val_volumes, patch_width=PATCH_WIDTH, depth=DEPTH_CHUNKS,
                             depth_stride=4, width_stride=4, 
                             max_shadow_fraction=0.05, max_per_volume=None,
                             include_slow_axis=False)

gc.collect()

print(f"Train slabs: {len(train_indices):,}")
print(f"Val slabs:   {len(val_indices):,}")

# ============================================
#           Data Generators
# ============================================
train_gen = DataGenerator3D(
    volumes=train_volumes,
    valid_indices=train_indices,
    batch_size=BATCH_SIZE,
    outshape=PATCH_SHAPE,
    n_augmented_copies=N_AUGMENTED_COPIES,
)
val_gen = DataGenerator3D(
    volumes=val_volumes,
    valid_indices=val_indices,
    batch_size=BATCH_SIZE,
    outshape=PATCH_SHAPE,
    n_augmented_copies=0,
)
print(f"Train batches/epoch: {len(train_gen)}  |  Val batches/epoch: {len(val_gen)}")
print(f"Number of augmented copies per batch: {N_AUGMENTED_COPIES}")

actual_steps_per_epoch = min(STEPS_PER_EPOCH, len(train_gen))
total_steps = EPOCHS * actual_steps_per_epoch


# ============================================
#               Models
# ============================================
def build_unet(shape=PATCH_SHAPE):
    """
    3D U-Net for vessel shadow inpainting.

    Strides are (1, 2, 2) throughout: spatial downsampling only.  The depth
    axis is kept at full resolution so inter-slice context is available at every level.

    Encoder:    (D, 256, 64) -> (D, 128, 32) -> (D, 64, 16) -> (D, 32, 8)
    Bottleneck: (D, 32, 8)
    Decoder:    (D, 32, 8)  -> (D, 64, 16)  -> (D, 128, 32) -> (D, 256, 64)

    (3, 3, 3) kernels allow the network to learn cross-B-scan patterns.
    """
    inputs = tf.keras.Input(shape=shape)    # (D, H, W, 1)

    # ==== Encoder ====
    # Level 1: (D, 256, 64) -> skip c1, stride -> (D, 128, 32)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    c1 = layers.Dropout(0.1)(c1)
    p1 = layers.Conv3D(32, (1, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(c1)

    # Level 2: (D, 128, 32) -> skip c2, stride -> (D, 64, 16)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    c2 = layers.Dropout(0.1)(c2)
    p2 = layers.Conv3D(64, (1, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(c2)

    # Level 3: (D, 64, 16) -> skip c3, stride -> (D, 32, 8)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    c3 = layers.Dropout(0.1)(c3)
    p3 = layers.Conv3D(128, (1, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(c3)

    # ==== Bottleneck: (D, 32, 8) ====
    bn = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    bn = layers.BatchNormalization()(bn)
    bn = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(bn)
    bn = layers.Dropout(0.1)(bn)

    # ==== Decoder ====
    # Level 3: (D, 32, 8) -> (D, 64, 16)
    u3 = layers.UpSampling3D(size=(1, 2, 2))(bn)
    u3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u3)
    d3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(d3)
    d3 = layers.Dropout(0.1)(d3)

    # Level 2: (D, 64, 16) -> (D, 128, 32)
    u2 = layers.UpSampling3D(size=(1, 2, 2))(d3)
    u2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u2)
    d2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(d2)
    d2 = layers.Dropout(0.1)(d2)

    # Level 1: (D, 128, 32) -> (D, 256, 64)
    u1 = layers.UpSampling3D(size=(1, 2, 2))(d2)
    u1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u1)
    d1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(d1)

    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(d1)
    return tf.keras.Model(inputs, outputs, name='unet_3d_inpainter')

def make_discriminator_model():
    """
    Conditional 3D discriminator that evaluates the cropped synthetic-shadow
    region of a volume chunk. The discriminator receives the image crop and
    its corresponding binary mask as two input channels.
    """
    model = tf.keras.Sequential()

    model.add(layers.Conv3D(64,  (3, 5, 5), strides=(1, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (3, 5, 5), strides=(1, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(256, (3, 5, 5), strides=(1, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.Dense(1))

    return model


autoencoder  = build_unet(PATCH_SHAPE)
discriminator = make_discriminator_model()


# ============================================
#               Loss & Optimizers
# ============================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def weighted_pixel_mse_3d(real, recon, mask, unmasked_weight=UNMASKED_WEIGHT):
    weight_map = mask + unmasked_weight * (1.0 - mask)
    sq_err = tf.square(real - recon)
    return tf.reduce_sum(weight_map * sq_err) / tf.reduce_sum(weight_map)

# Shared ResNet feature extractor (content + style loss) 
def build_feature_extractor():
    """
    Frozen ResNet152 feature extractor used by both content and style losses.
    ResNet152 stage block counts: 3 / 8 / 36 / 3
      conv2_block3_out   = layer 9   (256 ch, stride 4)
      conv3_block8_out   = layer 33  (512 ch, stride 8)
      conv4_block36_out  = layer 141 (1024 ch, stride 16)
      conv5_block3_out   = layer 150 (2048 ch, stride 32)
    """
    weights_path = os.path.join(os.path.dirname(__file__),
                                'resnet', 'resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base = tf.keras.applications.ResNet152(
        include_top=False,
        weights=weights_path,
        input_shape=(256, None, 3),  # None allows variable width for cropped regions
    )
    base.trainable = False
    layer_names = [
        'conv2_block3_out',    # layer 9
        'conv3_block8_out',    # layer 33
        'conv4_block36_out',   # layer 141
        'conv5_block3_out',    # layer 150
    ]
    outputs = [base.get_layer(n).output for n in layer_names]
    return tf.keras.Model(base.input, outputs, name='feature_extractor')

feature_extractor = build_feature_extractor()  

def extract_masked_region_3d(image, mask):
    """
    Crop the full 3D chunk to the shadow mask bounding box.
    image: (B, D, H, W, C)
    mask:  (B, D, H, W, C)
    """
    col_sums     = tf.reduce_sum(mask, axis=[0, 1, 2, 4])   
    nonzero_cols = tf.where(col_sums > 0)[:, 0]
    start_col    = tf.reduce_min(nonzero_cols)
    end_col      = tf.reduce_max(nonzero_cols) + 1
    return image[:, :, :, start_col:end_col, :]

def get_features_3d(image_3d):
    """
    Computes ResNet features for the entire 3D chunk by flattening 
    the Depth axis into the Batch axis to pass through the 2D ResNet.
    image_3d: (B, D, H, W, 1)
    """
    B = tf.shape(image_3d)[0]
    D = tf.shape(image_3d)[1]
    H = tf.shape(image_3d)[2]
    W = tf.shape(image_3d)[3]
    
    flat_2d = tf.reshape(image_3d, [B * D, H, W, 1])
    rgb = tf.tile(flat_2d, [1, 1, 1, 3]) * 255.0
    pre = tf.keras.applications.resnet.preprocess_input(rgb)
    return feature_extractor(pre, training=False)

# GRAM MATRIX: computes the CxC Gram matrix from a feature map

def gram_matrix_features(feat):
    """
    CxC Gram matrix from a ResNet feature map.
    feat: (B, H, W, C)
    Returns: (B, C, C) normalised by C only
    """
    B = tf.shape(feat)[0]
    H = tf.shape(feat)[1]
    W = tf.shape(feat)[2]
    C = tf.shape(feat)[3]
    F = tf.reshape(feat, [B, H * W, C])          # (B, H*W, C)
    G = tf.matmul(F, F, transpose_a=True)        # (B, C, C)
    return G / tf.cast(C, tf.float32)            # Changed: divide by C only

# DISCRIMINATOR LOSS: binary cross-entropy loss for real vs fake classification.

def discriminator_loss(real_output, fake_output):
    return (cross_entropy(tf.ones_like(real_output), real_output) +
            cross_entropy(tf.zeros_like(fake_output), fake_output))

# AUTOENCODER LOSS: combined loss for the generator, including pixel-wise MSE, adversarial loss,
# style loss, and content loss.

def autoencoder_loss(real_3d, recon_3d, fake_output, mask_3d,
                     unmasked_weight=UNMASKED_WEIGHT, rec_weight=REC_WEIGHT, 
                     adv_weight=ADV_WEIGHT, style_weight=STYLE_WEIGHT, 
                     content_weight=CONTENT_WEIGHT):
                     
    real_3d  = tf.cast(real_3d,  tf.float32)
    recon_3d = tf.cast(recon_3d, tf.float32)
    mask_3d  = tf.cast(mask_3d,  tf.float32)

    # 1. Pixel loss computed on FULL 3D volume
    pixel_loss = weighted_pixel_mse_3d(real_3d, recon_3d, mask_3d, unmasked_weight)
    
    # 2. Adversarial loss from the FULL 3D chunk
    adv_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # 3. Perceptual loss computed on ALL slices in the 3D chunk
    real_masked  = extract_masked_region_3d(real_3d,  mask_3d)
    recon_masked = extract_masked_region_3d(recon_3d, mask_3d)
    
    real_feats   = get_features_3d(real_masked)
    recon_feats  = get_features_3d(recon_masked)

    # CONTENT LOSS: mean squared error between real vs recon features

    c_losses = []
    for rf, rcf in zip(real_feats, recon_feats):
        c_losses.append(tf.reduce_mean(tf.square(rf - rcf)))
    c_loss = tf.add_n(c_losses)

    # STYLE LOSS: mean squared error between Gram matrices of real vs recon features

    s_losses = []
    for rf, rcf in zip(real_feats, recon_feats):
        G_real  = gram_matrix_features(rf)
        G_recon = gram_matrix_features(rcf)
        s_losses.append(tf.reduce_mean(tf.square(G_real - G_recon)))
    s_loss = tf.add_n(s_losses)

    total = (rec_weight * pixel_loss + adv_weight * adv_loss
             + style_weight * s_loss + content_weight * c_loss)
    return total, pixel_loss, adv_loss, s_loss, c_loss

ae_optimizer   = tf.keras.optimizers.Adam(learning_rate=AE_LR_START)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=DISC_LR_START)

# ============================================
#               Checkpoints
# ============================================
checkpoint_dir = f'./checkpoints/{RUN_NAME}'
os.makedirs(checkpoint_dir, exist_ok=True)

trained_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)
best_val_ae = tf.Variable(best_val_ae_initial, dtype=tf.float32, trainable=False)
epochs_without_improve_lr = tf.Variable(0, dtype=tf.int64, trainable=False)
epochs_without_improve_es = tf.Variable(0, dtype=tf.int64, trainable=False)
checkpoint = tf.train.Checkpoint(
    autoencoder_optimizer=ae_optimizer,
    discriminator_optimizer=disc_optimizer,
    autoencoder=autoencoder,
    discriminator=discriminator,
    trained_epochs=trained_epochs,
    best_val_ae=best_val_ae,
    epochs_without_improve_lr=epochs_without_improve_lr,
    epochs_without_improve_es=epochs_without_improve_es,
)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Best-by-ValAE checkpoint (keep only one)
best_checkpoint_dir = os.path.join(checkpoint_dir, "best")
os.makedirs(best_checkpoint_dir, exist_ok=True)
best_manager = tf.train.CheckpointManager(
    checkpoint=checkpoint,
    directory=best_checkpoint_dir,
    max_to_keep=1,
    checkpoint_name="best",
)

# ============================================
#            Adversarial Training
# ============================================
@tf.function
def train_step(masked, target, mask_3d):
    """
    masked:   (B, D, H, W, 1) — full 3D chunk with shadow zeroed across all D scans.
    target:   (B, D, H, W, 1) — clean full chunk.
    mask_3d:  (B, D, H, W, 1) — full angled shadow mask for all D B-scans.
    """
    masked   = tf.cast(masked,   tf.float32)
    target   = tf.cast(target,   tf.float32)
    mask_3d  = tf.cast(mask_3d,  tf.float32)

    with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
        recon = autoencoder(masked, training=True)   # (B, D, H, W, 1)

        # Discriminator sees only the masked region of full 3D chuck
        target_crop = extract_masked_region_3d(target, mask_3d)
        recon_crop  = extract_masked_region_3d(recon,  mask_3d)
        mask_crop   = extract_masked_region_3d(mask_3d, mask_3d)

        real_input = tf.concat([target_crop, mask_crop], axis=-1)
        fake_input = tf.concat([recon_crop,  mask_crop], axis=-1)
        
        real_out  = discriminator(real_input, training=True)
        fake_out  = discriminator(fake_input, training=True)

        ae_loss, px_loss, adv_loss, s_loss, c_loss = \
            autoencoder_loss(target, recon, fake_out, mask_3d)
            
        d_loss = discriminator_loss(real_out, fake_out)
        
    ae_optimizer.apply_gradients(
        zip(ae_tape.gradient(ae_loss, autoencoder.trainable_variables),
            autoencoder.trainable_variables))
    disc_optimizer.apply_gradients(
        zip(disc_tape.gradient(d_loss, discriminator.trainable_variables),
            discriminator.trainable_variables))
            
    return ae_loss, d_loss, px_loss, adv_loss, s_loss, c_loss

def _save_grid(masked, target, epoch, tag, n=5, save_dir='Generated_Images'):
    """Save a grid showing first / middle / last B-scan for each example."""
    os.makedirs(save_dir, exist_ok=True)
    recon  = autoencoder(tf.cast(masked[:n], tf.float32), training=False).numpy()
    D      = masked.shape[1]
    d_idxs = [0, D // 2, D - 1]
    labels = [f"B-scan 0", f"B-scan {D//2} (mid)", f"B-scan {D-1}"]
 
    n_rows = len(d_idxs) * 3
    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows))
    for row_group, (d, lbl) in enumerate(zip(d_idxs, labels)):
        r0 = row_group * 3
        for i in range(n):
            axes[r0,     i].imshow(masked[i, d].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[r0,     i].axis('off')
            axes[r0 + 1, i].imshow(target[i, d].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[r0 + 1, i].axis('off')
            axes[r0 + 2, i].imshow(recon[i,  d].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[r0 + 2, i].axis('off')
        axes[r0,     0].set_ylabel(f"{lbl}\nMasked",  fontsize=7)
        axes[r0 + 1, 0].set_ylabel(f"{lbl}\nTarget",  fontsize=7)
        axes[r0 + 2, 0].set_ylabel(f"{lbl}\nRecon",   fontsize=7)
    plt.suptitle(f"Epoch {epoch} — {tag}  (first / mid / last B-scan of chunk)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{RUN_NAME}_{tag}_epoch_{epoch:03d}.png"))
    plt.close()
 
def _spread_samples_3d(gen, n=5):
    """Pick n batches evenly spread across the generator, one chunk from each."""
    indices = np.linspace(0, len(gen) - 1, n, dtype=int)
    masked_list, target_list = [], []
    for idx in indices:
        m, t, _ = gen[idx]
        masked_list.append(m[0])    # (D, H, W, 1)
        target_list.append(t[0])
    masked_out = np.stack(masked_list, axis=0)   # (n, D, H, W, 1)
    target_out = np.stack(target_list, axis=0)
    return masked_out, target_out
 
def save_samples(epoch, n=5, save_dir='Generated_Images'):
    masked, target = _spread_samples_3d(val_gen, n=n)
    _save_grid(masked, target, epoch, tag='val', n=n, save_dir=save_dir)
 
def save_train_samples(epoch, n=5, save_dir='Generated_Images'):
    masked, target = _spread_samples_3d(train_gen, n=n)
    _save_grid(masked, target, epoch, tag='train', n=n, save_dir=save_dir)

def _lr_change_points(epochs, lrs):
    changes = []
    prev_lr = None

    for epoch, lr in zip(epochs, lrs):
        if prev_lr is None:
            prev_lr = lr
            continue
        if lr != prev_lr:
            changes.append((epoch, lr))
            prev_lr = lr

    return changes


def _add_lr_markers(ax, lr_changes):
    for epoch, lr in lr_changes:
        ax.axvline(epoch, color='black', linestyle='--', alpha=0.35, linewidth=1.2)
        ymax = ax.get_ylim()[1]
        ax.text(
            epoch,
            ymax,
            f"LR {lr:.2e}",
            rotation=90,
            va='top',
            ha='right',
            fontsize=8,
            alpha=0.75,
        )


resume_ckpt = tf.train.latest_checkpoint(best_checkpoint_dir)
if resume_ckpt is None:
    resume_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

if resume_ckpt:
    checkpoint.restore(resume_ckpt).expect_partial()
    start_epoch = int(trained_epochs.numpy())
    print(f"Resuming from checkpoint at epoch {start_epoch} ({resume_ckpt})")
    print(f"Restored AE LR:   {float(ae_optimizer.learning_rate.numpy()):.2e}")
    print(f"Restored Disc LR: {float(disc_optimizer.learning_rate.numpy()):.2e}")
    print(f"Restored best ValAE: {float(best_val_ae.numpy()):.4f}")
    print(f"Restored LR patience count: {int(epochs_without_improve_lr.numpy())}")
    print(f"Restored ES patience count: {int(epochs_without_improve_es.numpy())}")
else:
    start_epoch = 0

# Loss history for plotting 
history = {
    'epoch': [],
    'ae': [],
    'pixel': [],
    'adv': [],
    'style': [],
    'content': [],
    'disc': [],
    'val_ae': [],
    'val_px': [],
    'lr': [],
}

print(f"\n=== Adversarial Training: {RUN_NAME} ({EPOCHS} epochs) ===")
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    ae_losses, d_losses, px_losses, adv_losses, s_losses, c_losses = [], [], [], [], [], []

    # steps = len(train_gen)
    steps = min(STEPS_PER_EPOCH, len(train_gen))
    for i in range(steps):
        m, t, mk = train_gen[i]
        ae_l, d_l, px_l, adv_l, s_l, c_l = train_step(m, t, mk)
        ae_losses.append(ae_l.numpy())
        d_losses.append(d_l.numpy())
        px_losses.append(px_l.numpy())
        adv_losses.append(adv_l.numpy())
        s_losses.append(s_l.numpy())
        c_losses.append(c_l.numpy())

    # Validation - full AE objective over the full 3D volume
    val_ae_l, val_px_l, val_adv_l, val_s_l, val_c_l = [], [], [], [], []
    for i in range(len(val_gen)):
        vm, vt, vk = val_gen[i]
        vm = tf.cast(vm, tf.float32)
        vt = tf.cast(vt, tf.float32)
        vk = tf.cast(vk, tf.float32)

        vr = autoencoder(vm, training=False)

        vt_crop = extract_masked_region_3d(vt, vk)
        vr_crop = extract_masked_region_3d(vr, vk)
        vk_crop = extract_masked_region_3d(vk, vk)

        fake_input = tf.concat([vr_crop, vk_crop], axis=-1)
        fake_out = discriminator(fake_input, training=False)

        v_ae, v_px, v_adv, v_s, v_c = autoencoder_loss(vt, vr, fake_out, vk)

        val_ae_l.append(v_ae.numpy())
        val_px_l.append(v_px.numpy())
        val_adv_l.append(v_adv.numpy())
        val_s_l.append(v_s.numpy())
        val_c_l.append(v_c.numpy())

    mean_val_ae = float(np.mean(val_ae_l))
    mean_val_px = float(np.mean(val_px_l))

    # Learning rate scheduling + best-checkpoint tracking based on validation AE loss
    if mean_val_ae < float(best_val_ae.numpy()) - MIN_DELTA:
        best_val_ae.assign(mean_val_ae)
        epochs_without_improve_lr.assign(0)
        epochs_without_improve_es.assign(0)

        trained_epochs.assign(epoch + 1)
        best_path = best_manager.save()
        print(f"  Best checkpoint updated: {best_path} | ValAE={float(best_val_ae.numpy()):.4f}")
    else:
        epochs_without_improve_lr.assign_add(1)
        epochs_without_improve_es.assign_add(1)

        if int(epochs_without_improve_lr.numpy()) >= LR_PATIENCE:
            old_ae_lr = float(ae_optimizer.learning_rate.numpy())
            old_disc_lr = float(disc_optimizer.learning_rate.numpy())
            new_ae_lr = max(old_ae_lr * LR_FACTOR, MIN_LR)
            new_disc_lr = max(old_disc_lr * LR_FACTOR, MIN_LR)

            if new_ae_lr < old_ae_lr:
                ae_optimizer.learning_rate.assign(new_ae_lr)
                print(f"  AE learning rate reduced: {old_ae_lr:.2e} -> {new_ae_lr:.2e}")

            if new_disc_lr < old_disc_lr:
                disc_optimizer.learning_rate.assign(new_disc_lr)
                print(f"  Disc learning rate reduced: {old_disc_lr:.2e} -> {new_disc_lr:.2e}")

            epochs_without_improve_lr.assign(0)

        if int(epochs_without_improve_es.numpy()) >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}! "
                  f"Val AE loss has not improved for {EARLY_STOP_PATIENCE} epochs.")
            checkpoint.save(file_prefix=checkpoint_prefix)
            save_samples(epoch + 1)
            save_train_samples(epoch + 1)
            break

    if (epoch + 1) % 10 == 0:
        save_samples(epoch + 1)
        save_train_samples(epoch + 1)
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        trained_epochs.assign(epoch + 1)
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"  Checkpoint saved at epoch {epoch + 1}")

    history['epoch'].append(epoch + 1)
    history['ae'].append(np.mean(ae_losses))
    history['pixel'].append(np.mean(px_losses))
    history['adv'].append(np.mean(adv_losses))
    history['style'].append(np.mean(s_losses))
    history['content'].append(np.mean(c_losses))
    history['disc'].append(np.mean(d_losses))
    history['val_ae'].append(mean_val_ae)
    history['val_px'].append(mean_val_px)
    history['lr'].append(float(ae_optimizer.learning_rate.numpy()))

    print(f"Epoch {epoch+1}: AE={np.mean(ae_losses):.4f} Pixel={np.mean(px_losses):.4f} "
          f"Adv={np.mean(adv_losses):.4f} Style={np.mean(s_losses):.6f} "
          f"Content={np.mean(c_losses):.4f} Disc={np.mean(d_losses):.4f} "
          f"ValAE={mean_val_ae:.4f} ValPx={mean_val_px:.4f} "
          f"LR={float(ae_optimizer.learning_rate.numpy()):.2e} "
          f"({time.time()-t0:.1f}s)")

    train_gen.on_epoch_end()

# Final save
if EPOCHS % 25 != 0:
    checkpoint.save(file_prefix=checkpoint_prefix)
save_samples(EPOCHS)
save_train_samples(EPOCHS)
print(f"\nTraining complete. Checkpoint saved to {checkpoint_dir}")
print(f"Best checkpoint: {best_manager.latest_checkpoint} | Best ValAE={float(best_val_ae.numpy()):.4f}")

if history['epoch']:
    os.makedirs('Generated_Images', exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'{RUN_NAME} — Training Losses', fontsize=16)

    lr_changes = _lr_change_points(history['epoch'], history['lr'])
    plots = [
        (axes[0, 0], [('Train AE', history['ae'], 'tab:blue'), ('Val AE', history['val_ae'], 'tab:orange')], 'AE Loss'),
        (axes[0, 1], [('Pixel', history['pixel'], 'tab:green')], 'Pixel Loss'),
        (axes[0, 2], [('Adv', history['adv'], 'tab:red')], 'Adversarial Loss'),
        (axes[1, 0], [('Style', history['style'], 'tab:purple')], 'Style Loss'),
        (axes[1, 1], [('Content', history['content'], 'tab:brown')], 'Content Loss'),
        (axes[1, 2], [('Disc', history['disc'], 'tab:pink')], 'Discriminator Loss'),
        (axes[2, 0], [('Val AE', history['val_ae'], 'tab:orange')], 'Validation AE Loss'),
        (axes[2, 1], [('Val Pixel', history['val_px'], 'tab:cyan')], 'Validation Pixel Loss'),
        (axes[2, 2], [('LR', history['lr'], 'tab:gray')], 'Learning Rate'),
    ]

    for ax, series_list, title in plots:
        for label, values, color in series_list:
            ax.plot(history['epoch'], values, label=label, color=color, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        _add_lr_markers(ax, lr_changes)

    plt.tight_layout()
    plt.savefig(f'Generated_Images/{RUN_NAME}_loss_history.png', dpi=150)
    plt.close()
    print(f"  Loss plot: Generated_Images/{RUN_NAME}_loss_history.png")