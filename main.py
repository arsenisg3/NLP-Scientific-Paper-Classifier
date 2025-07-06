from pipeline import MLPipeline
import logging
import os
import sys

from constants import Constants as C

LOG_FILE_PATH = os.path.join(C.OUTPUT_DIR, 'pipeline_run.log')
os.makedirs(C.OUTPUT_DIR, exist_ok=True)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure File Handler
file_handler_exists = False
for handler in root_logger.handlers:
    if isinstance(handler, logging.FileHandler) and handler.baseFilename == LOG_FILE_PATH:
        file_handler_exists = True
        break

if not file_handler_exists:
    try:
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.getLogger(__name__).info(f"File logging set up to: {LOG_FILE_PATH}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to set up file logging to {LOG_FILE_PATH}: {e}")


# Configure Console Handler
console_handler_exists = False
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        console_handler_exists = True
        break

if not console_handler_exists:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.getLogger(__name__).info("Console logging set up.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    pipeline = MLPipeline()
    pipeline.run()


