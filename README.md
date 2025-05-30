# Dự án Phát hiện Ung thư Da

Dự án này triển khai một mô hình học sâu sử dụng mạng nơ-ron tích chập (CNN) dựa trên DenseNet201 để phân loại 9 loại ung thư da từ hình ảnh da liễu trong bộ dữ liệu ISIC (International Skin Imaging Collaboration). Mô hình được thiết kế để hỗ trợ nghiên cứu và học thuật, cung cấp một baseline mạnh mẽ cho các ứng dụng chẩn đoán y khoa bằng AI.

## Mục đích
- Hỗ trợ chẩn đoán ung thư da thông qua phân loại hình ảnh da liễu.
- Cung cấp một pipeline mô-đun, dễ tái sử dụng cho các bài toán phân loại hình ảnh.
- Tương thích với Google Colab, Kaggle, hoặc các môi trường TensorFlow có hỗ trợ GPU.

## Bộ dữ liệu
Dự án sử dụng bộ dữ liệu ISIC, bao gồm hình ảnh da liễu được phân loại thành 9 lớp bệnh:
1. Pigmented Benign Keratosis
2. Melanoma
3. Vascular Lesion
4. Actinic Keratosis
5. Squamous Cell Carcinoma
6. Basal Cell Carcinoma
7. Seborrheic Keratosis
8. Dermatofibroma
9. Nevus

Dữ liệu được tiền xử lý với kích thước ảnh 128x128 (có thể điều chỉnh qua file cấu hình).

## Hiệu quả mô hình
- **Mean AUC**: 0.99
- **Độ chính xác trên tập kiểm tra**: 92%
- **Đánh giá chi tiết**: Precision, recall, và F1-score được cung cấp cho từng lớp bệnh.
- **Trực quan hóa**: Ma trận nhầm lẫn, đường cong ROC, đường cong Precision-Recall, và dự đoán trên các mẫu ngẫu nhiên.

## Cấu trúc thư mục
- `config/`: Chứa file cấu hình `config.yaml` để quản lý tham số (đường dẫn dữ liệu, kích thước ảnh, v.v.).
- `src/`: Chứa mã nguồn Python chính:
  - `data_preprocessing.py`: Tiền xử lý và cân bằng dữ liệu.
  - `model.py`: Xây dựng mô hình CNN dựa trên DenseNet201.
  - `train.py`: Huấn luyện mô hình.
  - `evaluate.py`: Đánh giá mô hình (ma trận nhầm lẫn, ROC, Precision-Recall).
  - `visualize.py`: Trực quan hóa dữ liệu và kết quả.
  - `utils.py`: Các hàm tiện ích (kiểm tra phần cứng, tải cấu hình).
- `output/`: Thư mục lưu biểu đồ, mô hình, và log.
- `requirements.txt`: Danh sách thư viện phụ thuộc.
- `main.py`: Điểm vào chính để chạy pipeline.

## Yêu cầu phần cứng
- **GPU**: Khuyến nghị NVIDIA với CUDA/CUDNN để huấn luyện nhanh hơn.
- **RAM**: Tối thiểu 16GB cho dữ liệu lớn.
- **Python**: Phiên bản 3.11.

## Cài đặt
1. Cài đặt Python 3.11.
2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt


Tải bộ dữ liệu ISIC và cập nhật đường dẫn trong config/config.yaml.
(Tùy chọn) Tải mô hình đã huấn luyện từ Hugging Face: vanhai123/skin_cancer_detection.

Sử dụng

Chỉnh sửa tham số trong config/config.yaml (ví dụ: đường dẫn dữ liệu, kích thước ảnh, số epoch).
Chạy pipeline chính:python main.py


Kết quả (biểu đồ, mô hình, log) được lưu trong thư mục output/.

Sử dụng mô hình đã huấn luyện
Để sử dụng mô hình .h5 đã tải về:
from tensorflow.keras.models import load_model
import numpy as np

# Nạp mô hình
model = load_model("output/skin_cancer_model.h5")

# Tiền xử lý ảnh đầu vào (resize về 128x128 RGB)
image_tensor = ...  # Thêm code tiền xử lý ảnh
pred = model.predict(image_tensor)

Tùy chỉnh

Thay đổi tham số: Chỉnh sửa config.yaml để thay đổi đường dẫn dữ liệu, kích thước ảnh, hoặc các siêu tham số.
Thử nghiệm mô hình mới: Mở rộng model.py để tích hợp các kiến trúc CNN khác.
Thêm trực quan hóa: Bổ sung các hàm mới vào visualize.py để tạo thêm biểu đồ.
Tích hợp API: Sử dụng mô hình trong ứng dụng thông qua TensorFlow Serving hoặc Flask.

Logging

File log được lưu tại output/output.log để theo dõi quá trình thực thi và debug.

Giấy phép

Giấy phép: MIT License (cho phép sử dụng phi thương mại và trong học thuật).
Trích dẫn: Nếu sử dụng trong nghiên cứu, vui lòng ghi nhận tác giả và link tới repository: vanhai123/skin_cancer_detection.

Tác giả

Tên: Hà Văn Hải
Email: vanhai11203@gmail.com
Hugging Face: vanhai123

Liên hệ
Nếu bạn cần hỗ trợ, trao đổi, hoặc hợp tác nghiên cứu, hãy liên hệ qua email hoặc Hugging Face. Tôi rất mong nhận được phản hồi để cải thiện dự án!
Lưu ý

Đảm bảo môi trường có hỗ trợ GPU và cài đặt CUDA/CUDNN cho TensorFlow 2.15.0.
Dự án được thiết kế mô-đun để dễ dàng tái sử dụng trong các bài toán phân loại hình ảnh khác.
Kiểm tra file log (output/output.log) để xử lý lỗi nếu xảy ra.



