import mmap
import os
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_PATH = "/dev/wpi/weights/layer82"

def consume_weights():
    logger.info(f"Waiting for {FILE_PATH} to be created...")
    while not os.path.exists(FILE_PATH):
        time.sleep(2)
    
    logger.info(f"File {FILE_PATH} found. Attempting to mmap...")
    
    try:
        with open(FILE_PATH, "r+b") as f:
            # Memory-map the file
            mm = mmap.mmap(f.fileno(), 0)
            
            # Treat it as a numpy array of float16
            # We use dtype=np.float16 as requested
            weights = np.frombuffer(mm, dtype=np.float16)
            
            logger.info(f"Successfully mapped weights. Array shape: {weights.shape}")
            logger.info(f"First 10 elements: {weights[:10]}")
            logger.info(f"Array mean: {weights.mean()}")
            
            # Keep the pod alive for inspection if needed
            logger.info("Weight consumption successful. Keeping pod alive...")
            while True:
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"Error consuming weights: {e}")

if __name__ == "__main__":
    consume_weights()
