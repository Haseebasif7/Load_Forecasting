"""Training hyperparameters (strong setup + optional multi-seed ensemble)."""

WINDOW = 336  # hours
HORIZON = 24  # hours
FEATURE_DIM = 8  # scaled load + sin/cos hour + sin/cos dow + 3 DST flags
EPS_LOAD = 1e-3  # kW: leading-zero trim

HIDDEN_SIZE = 288  # marginal capacity bump vs 256; reduce if GPU OOM
LSTM_LAYERS = 2
DROPOUT = 0.25

USE_SEQUENCE_POOL = True

BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 22
GRAD_CLIP = 1.0

# Batch-wise OneCycle (replaces plateau on val loss): step every training batch.
ONE_CYCLE_PCT_START = 0.12

# Seeds saved as ensemble_member_seed{seed}.pt; eval averages logits in scaled space.
# Set tuple of one seed for fastest training (same as before syntactically).
ENSEMBLE_SEEDS: tuple[int, ...] = (42, 123, 777)

HORIZON_LOSS_WEIGHT_START = 1.28
HORIZON_LOSS_WEIGHT_END = 0.72

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # next 10%; test = last 10%

ZIP_MEMBER = "LD2011_2014.txt"
RAW_ZIP_RELPATH = "raw/electricityloaddiagrams20112014.zip"

PROCESSED_DIR = "processed"
MODELS_DIR = "models"
