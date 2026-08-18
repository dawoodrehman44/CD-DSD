import os
import torch


class Config:

    # =========================================================================
    # DATASET PATHS
    # =========================================================================

    CHEXPERT_IMAGE_ROOT = "/mnt/Internal/MedImage/unzip_chexpert_images/CheXpert-v1.0/train/"
    CHEXPERT_TRAIN_CSV  = "/mnt/Internal/MedImage/chexpert_balanced_for_training_3000_per_label_dis+demog+age.csv"
    CHEXPERT_VALID_CSV  = "/mnt/Internal/MedImage/chexpert_balanced_for_training_252_per_label_dis+demog+age.csv"

    MIMIC_IMAGE_ROOT    = "/mnt/External/Seagate/dawood/datasets/mimic-cxr/jpg"
    MIMIC_TRAIN_CSV     = "/mnt/External/Seagate/dawood/datasets/mimic-cxr/cleaned/mimic_clean_train.csv"
    MIMIC_VALID_CSV     = "/mnt/External/Seagate/dawood/datasets/mimic-cxr/cleaned/mimic_clean_valid.csv"

    NIH_IMAGE_ROOT      = "/mnt/External/Seagate/dawood/datasets/Chest_Xray_08/ChestX-Ray8 dataset/Images/images"
    NIH_CSV             = "/mnt/External/Seagate/dawood/datasets/Chest_Xray_08/ChestX-Ray8 dataset/Data_Entry_2017_v2020.csv"

    # =========================================================================
    # LABELS
    # =========================================================================

    NUM_CLASSES = 8
    LABEL_COLS  = [
        'No Finding', 'Atelectasis', 'Cardiomegaly',
        'Consolidation', 'Edema', 'Pleural Effusion',
        'Pneumonia', 'Pneumothorax'
    ]

    # =========================================================================
    # OUTPUT DIRS
    # =========================================================================

    OUTPUT_DIR     = "/home/dawood/lab2_rotaion/counterfactual_diff_uncertainty/"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
    PLOT_DIR       = os.path.join(OUTPUT_DIR, "plots")
    RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")

    # =========================================================================
    # GENERAL
    # =========================================================================

    DEVICE     = "cuda:1" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = 224
    MEAN       = [0.485, 0.456, 0.406]
    STD        = [0.229, 0.224, 0.225]
    SEED       = 42

    # =========================================================================
    # DIFFUSION MODEL
    # =========================================================================

    DIFF_BASE_CHANNELS  = 64
    DIFF_CHANNEL_MULTS  = (1, 2, 4, 8)
    DIFF_DROPOUT        = 0.1
    DIFF_ATTN_DEPTHS    = (2, 3)
    DIFF_T              = 1000
    DIFF_BETA_START     = 1e-4
    DIFF_BETA_END       = 0.02
    DIFF_SCHEDULE       = "linear"

    # =========================================================================
    # CLASSIFIER
    # =========================================================================

    CLS_CKPT_NAME  = "classifier_best.pt"
    CLS_DROPOUT    = 0.3
    CLS_EPOCHS     = 50
    CLS_LR         = 1e-4
    CLS_BATCH_SIZE = 64
    CLS_GRAD_CLIP  = 1.0

    # =========================================================================
    # CD-DSD INFERENCE
    # =========================================================================

    MC_SAMPLES             = 30
    DOMAIN_SCORE_TIMESTEPS = 10
    DOMAIN_SCORE_REPEATS   = 4
    DDIM_STEPS             = 50
    DDIM_ETA               = 0.0
    T_STAR_FULL            = 200

    T_STAR_LEVELS = {
        "brightness_contrast": 50,
        "noise_texture":       150,
        "scanner_artefact":    300,
        "global_structure":    500,
    }

    # =========================================================================
    # HFER (High-Frequency Energy Ratio) — blur/resolution domain signal
    # =========================================================================
    HFER_RADIUS_FRAC = 0.25   # frequencies above this fraction of Nyquist = HF
    HFER_WEIGHT      = 1.0    # relative weight vs. diffusion denoising score
