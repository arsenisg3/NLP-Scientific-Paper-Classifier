from dataclasses import dataclass
import os


@dataclass
class Constants:
    """Dataclass to handle individual constants."""

    # Directories
    CURRENT_DIRECTORY = os.getcwd()

    # Cleaning parameters
    DATE_FORMAT = '%m/%d/%y'
    TIME_COLS = ['published_date', 'updated_date']
    JUNK_VALUES = ['#name?', 'n/a', 'null', 'test', '', 'none']
    MIN_WORDS = 20
    MAX_WORDS = 320
    DESIRED_CATEGORY_COUNT_THRESHOLD = 1000
    MAX_SAMPLES_PER_CATEGORY = 10000

    # Model
    MODEL_NAME = "distilbert-base-uncased"

    # Training parameters
    STEPS: int = 3000
    SEED: int = 42
    SPLIT_RATIO: float = 0.8
    OUTPUT_DIR: str = './results'
    NUM_TRAIN_EPOCHS: int = 1.5
    PER_DEVICE_BATCH_SIZE: int = 12
    DATALOADER_NUM_WORKERS: int = 2
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01
    LOGGING_DIR: str = './logs'
    LOGGING_STEPS: int = STEPS
    EVAL_STRATEGY: str = 'steps'
    EVAL_STEPS: int = STEPS
    SAVE_STRATEGY: str = 'steps'
    SAVE_STEPS: int = STEPS
    SAVE_TOTAL_LIMIT = None
    METRIC_FOR_BEST_MODEL: str = "eval_f1_macro"
    GREATER_IS_BETTER: bool = True
    LOAD_BEST_MODEL_AT_END: bool = True
    REPORT_TO: str = "none"

    # Plotting
    OUTPUT_PLOT_DIR: str = './plots'
    SAVEFIG: bool = True
