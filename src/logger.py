import logging
import os
from datetime import datetime

# Create a logs directory if not exists
LOG_DIR = "logs"
logs_path = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(logs_path, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create a logger instance
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.info("Logging has started")
