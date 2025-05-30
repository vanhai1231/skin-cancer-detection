# Skin Cancer Detection using CNN

Dự án này nhằm mục đích phát triển mô hình Deep Learning dựa trên CNN (Ảnh học Chuyển Tích) để phân loại 9 loại ung thư da, sử dụng bộ dữ liệu ISIC.

---

## 📁 Cấu trúc thư mục

```
skin-cancer-detection/
├── data/                            # Thư mục chứa dữ liệu (nếu có)
├── model/
│   └── skin_cancer_model.h5        # Mô hình đã huấn luyện lưu dưới dạng HDF5
├── results/                         # Kết quả và biểu đồ trực quan
│   ├── confusion_matrix.png
│   ├── data_distribution.png
│   ├── random_sample_predictions.png
│   ├── roc_curves.png
│   ├── sample_images.png
│   └── training_history.png
├── README.md                       # Tài liệu hướng dẫn (file này)
├── requirements.txt                # Danh sách thư viện Python cần cài
└── skin-cancer-detection.ipynb     # Notebook chính của dự án
```

---

## 📁 Dataset

* Tên: **Skin Cancer - 9 Class ISIC Dataset**
* Nguồn: [https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

---

## 🚀 Cách chạy dự án

### 1. Clone repository:

```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

### 3. Mở notebook trên Jupyter hoặc Google Colab:

* **Google Colab**: Tải file `skin-cancer-detection.ipynb` lên [https://colab.research.google.com](https://colab.research.google.com)
* Hoặc mở bằng lệnh:

```bash
jupyter notebook skin-cancer-detection.ipynb
```

---

## 🔎 Tổng quan quy trình mô hình CNN

### 1. SỬ DỤNG MÔ HÌNH CNN

* Import các thư viện cần thiết
* Kiểm tra GPU
* DataFrame, resize ảnh
* Hàm vẽ biểu đồ
* Hàm tạo & huấn luyện mô hình
* Nạp & resize toàn bộ ảnh
* Đường dẫn
* Tạo DataFrame
* Tạo `label_map`
* Giới hạn mỗi lớp
* Resize song song
* Loại bỏ các ảnh lỗi
* Chia dữ liệu
* Data Augmentation
* Tóm tắt & trực quan dữ liệu
* Chuẩn bị train/val/test
* Giải phóng bộ nhớ
* Huấn luyện mô hình
* Đánh giá trên tập Test
* Vẽ đường cong Precision–Recall
* Lưu mô hình & dự đoán một số mẫu

---

## Kết quả mô hình

* **Mean AUC**: 0.99
* **Accuracy (Tập test)**: 92%

### Báo cáo chi tiết:

| Class                      | Precision | Recall | F1-score | Support |
| -------------------------- | --------- | ------ | -------- | ------- |
| Pigmented Benign Keratosis | 0.97      | 0.93   | 0.95     | 390     |
| Melanoma                   | 0.87      | 0.73   | 0.79     | 422     |
| Vascular Lesion            | 1.00      | 1.00   | 1.00     | 385     |
| Actinic Keratosis          | 0.82      | 1.00   | 0.90     | 406     |
| Squamous Cell Carcinoma    | 0.96      | 0.98   | 0.97     | 401     |
| Basal Cell Carcinoma       | 0.97      | 0.97   | 0.97     | 399     |
| Seborrheic Keratosis       | 0.85      | 0.90   | 0.87     | 417     |
| Dermatofibroma             | 1.00      | 0.99   | 0.99     | 389     |
| Nevus                      | 0.87      | 0.78   | 0.82     | 391     |

**Tổng hợp:**

* Accuracy: 0.92
* Macro avg: 0.92
* Weighted avg: 0.92

**Hình ảnh minh hoạ:** nằm trong thư mục `results/`

* `confusion_matrix.png`
* `roc_curves.png`
* `training_history.png`
* `sample_images.png`, v.v.

---

## 🎓 Kiến thức áp dụng

* TensorFlow / Keras
* Xử lý ảnh: OpenCV, matplotlib
* Tiền xử lý dữ liệu: Pandas, Numpy
* Machine Learning metrics

---

## 🚨 Lưu ý

* Dự án dùng cho mục đích **nghiên cứu khoa học, phi thương mại**
* Mã nguồn có thể được tái sử dụng với trích dẫn nguồn thích hợp

---

## 📚 Bản quyền

Tác giả: [Hà Văn Hải](https://www.kaggle.com/haivan11)

Mã nguồn chia sẻ theo giấy phép MIT. Mọi người đều có thể dùng và tuỳ biến theo nhu cầu.
