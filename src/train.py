import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, X_train, y_train, X_val, y_val, config):
    """Huấn luyện mô hình với dữ liệu đã tiền xử lý.

    Args:
        model: Mô hình Keras.
        X_train, y_train: Dữ liệu và nhãn huấn luyện.
        X_val, y_val: Dữ liệu và nhãn kiểm tra.
        config (dict): Cấu hình chứa epochs, patience_lr, factor_lr, min_lr, patience_early_stopping.

    Returns:
        tuple: Mô hình đã huấn luyện và lịch sử huấn luyện.

    Raises:
        ValueError: Nếu dữ liệu không hợp lệ.
    """
    try:
        # Tạo Data Augmentation
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        
        # Tính class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Callbacks
        lr_red = ReduceLROnPlateau(
            monitor='val_accuracy', 
            patience=config['training']['patience_lr'],
            factor=config['training']['factor_lr'],
            min_lr=config['training']['min_lr'],
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=config['training']['patience_early_stopping'],
            restore_best_weights=True
        )
        
        # Huấn luyện mô hình
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=config['data']['batch_size']),
            epochs=config['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[lr_red, early_stopping],
            class_weight=class_weight_dict
        )
        
        logger.info("Hoàn tất huấn luyện mô hình.")
        return model, history
    
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        raise