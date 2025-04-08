# Global configuration can be set here or imported from a separate config file.
class CFG:
    seed            = 42
    debug           = False
    apex            = False
    print_freq      = 100
    num_workers     = 1

    # Directories
    OUTPUT_DIR      = './'
    train_datadir   = './data/train_audio'
    train_csv       = './data/train.csv'
    taxonomy_csv    = './data/taxonomy.csv'
    spectrogram_npy = './data/birdclef2025_melspec_5sec_256_256.npy'

    # Model parameters
    model_name      = 'efficientnet_b0'
    pretrained      = True
    in_channels     = 1

    # Data parameters
    LOAD_DATA       = True    # If False, process audio on-the-fly.
    FS              = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE    = (256, 256)
    N_FFT           = 1024
    HOP_LENGTH      = 512
    N_MELS          = 128
    FMIN            = 50
    FMAX            = 14000

    device          = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    epochs          = 10
    batch_size      = 32
    criterion       = 'BCEWithLogitsLoss'

    n_fold          = 5
    selected_folds  = [0]

    optimizer       = 'AdamW'
    lr              = 5e-4
    weight_decay    = 1e-5

    scheduler       = 'CosineAnnealingLR'
    min_lr          = 1e-6
    T_max           = epochs

    aug_prob        = 0.5
    mixup_alpha     = 0.5

    dataset         = 'birdclef_2025'
    preprocessing   = 'precomputed'  # Or "on_the_fly"