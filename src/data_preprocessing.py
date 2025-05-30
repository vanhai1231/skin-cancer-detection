import os
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
import multiprocessing
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dataframe(data_dir):
    """Tạo DataFrame từ thư mục chứa dữ liệu hình ảnh.

    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu.

    Returns:
        pd.DataFrame: DataFrame chứa đường dẫn ảnh và nhãn.

    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
    """
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại.")
        data = [
            {"image_path": os.path.join(data_dir, cls_name, fname), "label": idx}
            for idx, cls_name in enumerate(os.listdir(data_dir))
            for fname in os.listdir(os.path.join(data_dir, cls_name))
        ]
        logger.info(f"Tạo DataFrame từ {data_dir} với {len(data)} mẫu.")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Lỗi khi tạo DataFrame: {str(e)}")
        raise

def resize_image_array(image_path):
    """Resize ảnh về kích thước xác định.

    Args:
        image_path (str): Đường dẫn đến ảnh.

    Returns:
        np.ndarray: Mảng ảnh đã resize, hoặc None nếu lỗi.
    """
    try:
        return np.asarray(Image.open(image_path).resize((128, 128)))
    except Exception as e:
        logger.warning(f"Lỗi khi resize ảnh {image_path}: {str(e)}")
        return None

def load_and_preprocess_data(config):
    """Tải, tiền xử lý và cân bằng dữ liệu.

    Args:
        config (dict): Cấu hình chứa train_dir, test_dir, max_per_class, image_size.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, label_map.

    Raises:
        ValueError: Nếu dữ liệu không hợp lệ.
    """
    try:
        train_dir = config['data']['train_dir']
        test_dir = config['data']['test_dir']
        max_per_class = config['data']['max_per_class']
        
        # Tạo DataFrame
        df_train = create_dataframe(train_dir)
        df_test = create_dataframe(test_dir)
        df = pd.concat([df_train, df_test], ignore_index=True)
        label_map = {i: cls for i, cls in enumerate(os.listdir(train_dir))}
        num_classes = len(label_map)
        logger.info(f"Số lớp: {num_classes}, Label map: {label_map}")

        # Giới hạn số lượng ảnh mỗi lớp
        df = df.groupby('label').head(max_per_class).reset_index(drop=True)
        
        # Resize ảnh song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            df['image'] = list(executor.map(resize_image_array, df['image_path']))
        
        # Loại bỏ ảnh lỗi
        df = df.dropna(subset=['image']).reset_index(drop=True)
        logger.info(f"Số mẫu sau khi loại bỏ lỗi: {len(df)}")

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.5,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        aug_df = pd.DataFrame(columns=['image_path', 'label', 'image'])
        for lbl in df['label'].unique():
            cls_df = df[df['label'] == lbl]
            aug_df = pd.concat([aug_df, cls_df], ignore_index=True)
            n_need = max_per_class - len(cls_df)
            if n_need > 0:
                imgs = cls_df['image'].values
                sel = np.random.choice(imgs, n_need, replace=True)
                for img in sel:
                    batch = np.expand_dims(img, 0)
                    aug_iter = datagen.flow(batch, batch_size=1)
                    aug = aug_iter.__next__()[0].astype('uint8')
                    new_row = pd.DataFrame([{'image_path': None, 'label': lbl, 'image': aug}])
                    aug_df = pd.concat([aug_df, new_row], ignore_index=True)
        
        # Cân bằng và shuffle
        df = aug_df.groupby('label').head(max_per_class).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Số mẫu sau khi cân bằng: {len(df)}")

        # Chuẩn bị dữ liệu
        X = np.stack(df['image'].values)
        y = to_categorical(df['label'].values, num_classes=num_classes)
        
        # Chuẩn hóa
        mean, std = X.mean(), X.std()
        if std == 0:
            raise ValueError("Dữ liệu có độ lệch chuẩn bằng 0, không thể chuẩn hóa.")
        X = (X - mean) / std
        
        # Chia dữ liệu
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        logger.info(f"Kích thước dữ liệu: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, label_map
    
    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
        raise
