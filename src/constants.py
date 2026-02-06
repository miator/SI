SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 30
FMIN = 20.0
FMAX = 8000.0

MAX_FRAMES = 300  # fixed time length after MFCC (pad/truncate)
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 1e-3

TRAIN_WAV_ROOT = r"C:\Users\User\Desktop\data\train"
TEST_WAV_ROOT = r"C:\Users\User\Desktop\data"
TEST_CSV = r"C:\Users\User\Desktop\data\test_full.csv"
