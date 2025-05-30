import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, label_map, output_dir):
    """Đánh giá mô hình trên tập kiểm tra và lưu kết quả.

    Args:
        model: Mô hình Keras đã huấn luyện.
        X_test: Dữ liệu kiểm tra.
        y_test: Nhãn kiểm tra.
        label_map: Ánh xạ từ chỉ số lớp sang tên nhãn.
        output_dir (str): Thư mục lưu kết quả.

    Returns:
        float: Giá trị AUC trung bình.

    Raises:
        ValueError: Nếu dữ liệu không hợp lệ.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        num_classes = len(label_map)
        
        # Ma trận nhầm lẫn
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title('Ma trận nhầm lẫn')
        plt.savefig(f'{output_dir}/confusion_matrix.png')
        plt.close()
        logger.info("Đã lưu ma trận nhầm lẫn.")
        
        # Báo cáo phân loại
        report = classification_report(y_test_classes, y_pred_classes, target_names=list(label_map.values()))
        logger.info(f"Báo cáo phân loại:\n{report}")
        
        # Đường cong ROC
        plt.figure(figsize=(12, 10))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{label_map[i]} (AUC={roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Tỉ lệ dương tính giả')
        plt.ylabel('Tỉ lệ dương tính đúng')
        plt.title('Đường cong ROC')
        plt.legend(loc='lower right')
        plt.savefig(f'{output_dir}/roc_curves.png')
        plt.close()
        mean_auc = np.mean(list(roc_auc.values()))
        logger.info(f"Mean AUC: {mean_auc:.2f}")
        
        # Đường cong Precision-Recall
        plt.figure(figsize=(12, 10))
        for i in range(num_classes):
            prec, rec, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            plt.plot(rec, prec, lw=2, label=f"{label_map[i]} (AP={auc(rec, prec):.2f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'{output_dir}/precision_recall_curves.png')
        plt.close()
        logger.info("Đã lưu đường cong Precision-Recall.")
        
        return mean_auc
    
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá mô hình: {str(e)}")
        raise
