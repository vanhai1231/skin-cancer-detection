import os
import numpy as np
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_data_distribution, plot_sample_images, plot_training_history, predict_random_samples
from src.utils import check_hardware, load_config
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='output/output.log')
logger = logging.getLogger(__name__)

def main():
    """Điểm vào chính để chạy pipeline phát hiện ung thư da."""
    try:
        # Tải cấu hình
        config = load_config('config/config.yaml')
        
        # Kiểm tra phần cứng
        check_hardware()
        
        # Tạo thư mục đầu ra
        output_dir = config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Tải và tiền xử lý dữ liệu
        X_train, X_val, X_test, y_train, y_val, y_test, label_map = load_and_preprocess_data(config)
        
        # Trực quan hóa dữ liệu
        counts = np.bincount(y_train.argmax(axis=1))
        plot_data_distribution(counts, label_map, output_dir)
        df = pd.DataFrame({'label': np.argmax(y_train, axis=1), 'image': list(X_train)})
        plot_sample_images(df, label_map, num_samples=9, output_dir=output_dir)
        
        # Xây dựng và huấn luyện mô hình
        model = build_model(config)
        model, history = train_model(model, X_train, y_train, X_val, y_val, config)
        
        # Vẽ lịch sử huấn luyện
        plot_training_history(history, output_dir)
        
        # Đánh giá mô hình
        mean_auc = evaluate_model(model, X_test, y_test, label_map, output_dir)
        
        # Dự đoán mẫu ngẫu nhiên
        predict_random_samples(model, X_test, y_test, label_map, num_samples=6, output_dir=output_dir)
        
        # Lưu mô hình
        model_path = f"{output_dir}/skin_cancer_model.h5"
        model.save(model_path)
        logger.info(f"Đã lưu mô hình tại {model_path}")
    
    except Exception as e:
        logger.error(f"Lỗi trong pipeline chính: {str(e)}")
        raise

if __name__ == "__main__":
    main()
