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
EPOCHS = 10
LEARNING_RATE = 1e-3

EMB_DIM = 192
EPS = 1e-6

BEST_MODEL_PATH = "runs/spk50_1h/triplet_bh_P12K5_m0.3_e10_cnn_spk50_best.pth"

DATA_ROOT = r"C:\Users\User\Desktop\50spk_1h\pcm16_16k_chunks_3s"
ALL_CHUNKS_CSV = r"C:\Users\User\Desktop\50spk_1h\all_chunks_csv"
