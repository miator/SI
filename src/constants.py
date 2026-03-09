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

EMB_DIM = 192
EPS = 1e-6

RESULTS_DIR = "results"

RUNS_DIR = "runs"
BEST_MODEL_PATH = f"{RUNS_DIR}/best.pt"

TRAIN_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\train"
VAL_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\val"
TEST_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\test"
