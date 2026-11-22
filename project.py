import os
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_CODE_DIR = "models"
DATASETS_CODE_DIR = "datasets"
THIRD_PARTY_DIR = "third_party"
CONFIG_FILE = "config.py"
PROJECT_FILE = "project.py"
TEST_FILE = "test.py"
TRAIN_FILE = "train.py"
TRAINER_FILE = "trainer.py"
EXP_SAVE_PATH = "exp"
EXP_CODE_DIR_AND_FILE = [
    os.path.join(PROJECT_ROOT_DIR, CONFIG_FILE),
    os.path.join(PROJECT_ROOT_DIR, DATASETS_CODE_DIR),
    os.path.join(PROJECT_ROOT_DIR, MODELS_CODE_DIR),
    os.path.join(PROJECT_ROOT_DIR, "misc"),
    os.path.join(PROJECT_ROOT_DIR, "moc_adapter.py"),
    os.path.join(PROJECT_ROOT_DIR, PROJECT_FILE),
    os.path.join(PROJECT_ROOT_DIR, TEST_FILE),
    os.path.join(PROJECT_ROOT_DIR, TRAIN_FILE),
    os.path.join(PROJECT_ROOT_DIR, TRAINER_FILE),
]
