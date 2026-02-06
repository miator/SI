SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 13
FMIN = 20.0
FMAX = 8000.0

MAX_FRAMES = 200  # fixed time length after MFCC (pad/truncate)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

TRAIN_WAV_ROOT = r"C:\Users\User\Desktop\data"
TEST_WAV_ROOT = r"C:\Users\User\Desktop\data"
