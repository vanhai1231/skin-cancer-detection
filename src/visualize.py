import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import array_to_img
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_training_history(history, output_dir):
    """Vẽ biểu đồ độ chính xác và hàm mất mát của quá trình huấn luyện.

    Args:
        history: Lịch sử huấn luyện từ model.fit().
        output_dir (str): Thư mục lưu biểu đồ.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history.history['accuracy'], label='Acc Huấn luyện')
        ax1.plot(history.history['val_accuracy'], label='Acc Kiểm tra')
        ax1.set_title('Độ chính xác')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history.history['loss'], label='Loss Huấn luyện')
        ax2.plot(history.history['val_loss'], label='Loss Kiểm tra')
        ax2.set_title('Hàm mất mát')
        ax2.legend()
        ax2.grid(True)
        plt.savefig(f'{output_dir}/training_history.png')
        plt.close()
        logger.info("Đã lưu biểu đồ lịch sử huấn luyện.")
    
    except Exception as e:
        logger.error(f"Lỗi khi vẽ lịch sử huấn luyện: {str(e)}")
        raise

def plot_data_distribution(class_counts, label_map, output_dir):
    """Vẽ biểu đồ phân bố số lượng ảnh theo lớp.

    Args:
        class_counts: Số lượng ảnh mỗi lớp.
        label_map: Ánh xạ từ chỉ số lớp sang tên nhãn.
        output_dir (str): Thư mục lưu biểu đồ.
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(list(label_map.values()), class_counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Phân bố số lượng ảnh theo lớp')
        plt.ylabel('Số ảnh')
        plt.savefig(f'{output_dir}/data_distribution.png')
        plt.close()
        logger.info("Đã lưu biểu đồ phân bố dữ liệu.")
    
    except Exception as e:
        logger.error(f"Lỗi khi vẽ phân bố dữ liệu: {str(e)}")
        raise

def plot_sample_images(df, label_map, num_samples, output_dir):
    """Vẽ các ảnh mẫu từ mỗi lớp.

    Args:
        df: DataFrame chứa dữ liệu ảnh và nhãn.
        label_map: Ánh xạ từ chỉ số lớp sang tên nhãn.
        num_samples (int): Số mẫu hiển thị.
        output_dir (str): Thư mục lưu biểu đồ.
    """
    try:
        plt.figure(figsize=(12, 12))
        for i, cls in enumerate(df['label'].unique()[:num_samples]):
            sample = df[df['label'] == cls].sample(1).iloc[0]
            plt.subplot(3, 3, i + 1)
            plt.imshow(sample['image'])
            plt.title(label_map[cls])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_images.png')
        plt.close()
        logger.info("Đã lưu ảnh mẫu.")
    
    except Exception as e:
        logger.error(f"Lỗi khi vẽ ảnh mẫu: {str(e)}")
        raise

def predict_random_samples(model, X_test, y_test, label_map, num_samples, output_dir):
    """Dự đoán và hiển thị kết quả trên các mẫu ngẫu nhiên từ tập kiểm tra.

    Args:
        model: Mô hình Keras đã huấn luyện.
        X_test: Dữ liệu kiểm tra.
        y_test: Nhãn kiểm tra.
        label_map: Ánh xạ từ chỉ số lớp sang tên nhãn.
        num_samples (int): Số mẫu ngẫu nhiên để hiển thị.
        output_dir (str): Thư mục lưu hình ảnh kết quả.
    """
    try:
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_samples = X_test[indices]
        y_samples = y_test[indices]
        
        y_pred = model.predict(X_samples)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_samples, axis=1)
        
        rows = (num_samples + 2) // 3
        plt.figure(figsize=(12, 4 * rows))
        
        for i in range(num_samples):
            plt.subplot(rows, 3, i + 1)
            img = array_to_img(X_samples[i])
            plt.imshow(img)
            plt.axis('off')
            
            true_label = label_map[y_true_classes[i]]
            pred_label = label_map[y_pred_classes[i]]
            pred_prob = y_pred[i][y_pred_classes[i]] * 100
            
            title = f"Thực tế: {true_label}\nDự đoán: {pred_label}\nXác suất: {pred_prob:.1f}%"
            plt.title(title, fontsize=10, pad=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/random_sample_predictions.png')
        plt.close()
        logger.info("Đã lưu dự đoán mẫu ngẫu nhiên.")
    
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán mẫu ngẫu nhiên: {str(e)}")
        raise
