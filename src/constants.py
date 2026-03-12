from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

EXP_NAME = "cnn1d_emb192_m025_P12K5_lr5e4_plateau"

RUNS_DIR = PROJECT_ROOT / "runs"
EXP_DIR = RUNS_DIR / EXP_NAME

TB_DIR = EXP_DIR / "tensorboard"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"

BEST_MODEL_PATH = CKPT_DIR / "best_val_loss.pt"
LAST_MODEL_PATH = CKPT_DIR / "last_epoch.pt"

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 30
FMIN = 20.0
FMAX = 8000.0

MAX_FRAMES = 301
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LR_SCHEDULER = "plateau"
LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-6

EMB_DIM = 192
EPS = 1e-6

TRIPLET_MARGIN = 0.25
P = 12
K = 5

TRAIN_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\train"
VAL_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\val"
TEST_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\test"

PRECOMPUTED_ROOT = Path(
    r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\logmel_cache"
)

TRAIN_FEAT_ROOT = PRECOMPUTED_ROOT / "train"
VAL_FEAT_ROOT = PRECOMPUTED_ROOT / "val"
TEST_FEAT_ROOT = PRECOMPUTED_ROOT / "test"

USE_PRECOMPUTED_FEATURES = True
