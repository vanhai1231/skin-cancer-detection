import tensorflow as tf
import multiprocessing
import yaml
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_hardware():
    """Kiểm tra GPU và CPU khả dụng."""
    try:
        logger.info(f"Số GPU khả dụng: {len(tf.config.list_physical_devices('GPU'))}")
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Số lõi CPU: {multiprocessing.cpu_count()}")
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra phần cứng: {str(e)}")
        raise

def load_config(config_path):
    """Tải cấu hình từ file YAML.

    Args:
        config_path (str): Đường dẫn đến file cấu hình.

    Returns:
        dict: Cấu hình đã tải.

    Raises:
        FileNotFoundError: Nếu file cấu hình không tồn tại.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Đã tải cấu hình từ {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"File cấu hình {config_path} không tồn tại.")
        raise
    except Exception as e:
        logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
        raise
