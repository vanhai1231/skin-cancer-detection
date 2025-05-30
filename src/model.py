
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import SGD
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_model(config):
    """Xây dựng mô hình CNN dựa trên DenseNet201.

    Args:
        config (dict): Cấu hình chứa input_shape, num_classes, learning_rate, momentum.

    Returns:
        Model: Mô hình Keras đã được cấu hình.

    Raises:
        ValueError: Nếu tham số cấu hình không hợp lệ.
    """
    try:
        input_shape = tuple(config['model']['input_shape'])
        num_classes = config['model']['num_classes']
        learning_rate = config['model']['learning_rate']
        momentum = config['model']['momentum']
        
        inp = Input(shape=input_shape)
        base = DenseNet201(include_top=False, weights='imagenet', input_tensor=inp)
        x = Flatten()(base.output)
        x = Dropout(0.7)(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inp, outputs=out)
        opt = SGD(learning_rate=learning_rate, momentum=momentum)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        logger.info(f"Đã xây dựng mô hình với input_shape={input_shape}, num_classes={num_classes}")
        return model
    
    except Exception as e:
        logger.error(f"Lỗi khi xây dựng mô hình: {str(e)}")
        raise
