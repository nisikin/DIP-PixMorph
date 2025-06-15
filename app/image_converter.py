import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QMessageBox,
)
from torch.cuda import device

# 导入图像处理模块
from src.basic.binarize import *
from src.basic.edge_detection import *
from src.basic.filtering import *
from src.basic.geometry import *
from src.basic.grayscale import *
from src.basic.histogram import *
from src.style_transfer.neural_style.run import *

class ImageConverterApp(QMainWindow):
    """图片转换工具的主窗口类，基于PyQt5实现图像处理GUI"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.original_image = None  # 原始图片
        self.processed_image = None  # 处理后的图片
        self.initUI()
        self.setWindowTitle("图片转换工具")  # 设置窗口标题
        self.setGeometry(100, 100, 900, 600)  # 设置窗口位置和大小

    def initUI(self):
        """初始化用户界面"""
        # 定义效果选项字典，包含各种图像处理类别及其子选项
        self.effect_options = {
            "灰度化": ["灰度化"],
            "二值化": ["二值化", "反二值化"],
            "图像平滑": [
                "均值滤波",
                "中值滤波",
                "理想低通滤波",
                "巴特沃斯低通滤波",
                "高斯低通滤波",
            ],
            "图像锐化": [
                "Roberts算子",
                "Sobel算子",
                "Prewitt算子",
                "Laplacian算子",
                "理想高通滤波",
                "巴特沃斯高通滤波",
                "高斯高通滤波",
            ],
            "绘制直方图": ["绘制直方图"],
            "直方图均衡化": ["对数变换", "均衡化", "正规化"],
            "几何变换": ["图像缩放", "图像旋转", "图像平移", "图像翻转", "仿射变换"],
            "边缘检测": [
                "Roberts算子边缘检测",
                "Prewitt算子边缘检测",
                "Sobel算子边缘检测",
                "Laplacian算子边缘检测",
                "LoG边缘检测",
                "Canny边缘检测",
                "霍夫变换直线检测",
            ],
            "风格迁移": ["1", "2", "3", "4"],
        }

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 顶部控制区域（水平布局）
        main_layout = QVBoxLayout()

        # 顶部控制区域
        control_layout = QHBoxLayout()

        # 文件选择按钮
        self.select_button = QPushButton("选择图片")
        self.select_button.setFixedHeight(50)
        self.select_button.clicked.connect(self.select_image)
        control_layout.addWidget(self.select_button)

        # 效果选择下拉框
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(self.effect_options.keys()))
        self.category_combo.setFixedHeight(20)
        self.category_combo.currentIndexChanged.connect(self.update_effect_combo)

        # 具体效果下拉框
        self.detail_combo = QComboBox()
        self.detail_combo.setFixedHeight(20)

        # 初始填充具体效果下拉框
        self.update_effect_combo(0)

        # 添加到布局
        effect_layout = QVBoxLayout()
        effect_layout.addWidget(self.category_combo)
        effect_layout.addWidget(self.detail_combo)
        control_layout.addLayout(effect_layout)

        # 处理按钮
        self.process_button = QPushButton("应用效果")
        self.process_button.setFixedHeight(50)
        self.process_button.clicked.connect(self.process_image)
        control_layout.addWidget(self.process_button)

        # 保存按钮
        self.save_button = QPushButton("保存结果")
        self.save_button.setFixedHeight(50)
        self.save_button.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_button)

        # 图片显示区域
        image_layout = QHBoxLayout()

        # 原始图片显示
        original_group = QGroupBox("原始图片")
        original_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)
        image_layout.addWidget(original_group)

        # 处理后的图片显示
        processed_group = QGroupBox("转换结果")
        processed_layout = QVBoxLayout()
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        self.processed_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        processed_layout.addWidget(self.processed_label)
        processed_group.setLayout(processed_layout)
        image_layout.addWidget(processed_group)

        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(image_layout)

        main_widget.setLayout(main_layout)

        # 状态信息
        self.status_label = QLabel("准备就绪，请选择图片")
        self.statusBar().addWidget(self.status_label)

        # 添加提示文本
        self.original_label.setText("点击上方按钮选择图片")
        self.processed_label.setText("选择效果后点击应用按钮")

    def update_effect_combo(self, index):
        """更新具体效果下拉框内容，根据类别选择动态填充"""
        category = self.category_combo.currentText()
        self.detail_combo.clear()
        self.detail_combo.addItems(self.effect_options.get(category, []))

    def select_image(self):
        """选择图片文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif);;所有文件 (*)",
            options=options,
        )

        if file_path:
            self.status_label.setText(f"已选择: {os.path.basename(file_path)}")
            # 加载图片
            self.original_image = QImage(file_path)
            if not self.original_image.isNull():
                pixmap = QPixmap.fromImage(self.original_image)
                self.original_label.setPixmap(
                    pixmap.scaled(
                        self.original_label.width(),
                        self.original_label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                self.processed_label.clear()
                self.processed_label.setText("选择效果后点击应用按钮")
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件")

    def process_image(self):
        """应用选定的图片处理效果"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return

        # 获取选定的效果
        effect = self.detail_combo.currentText()

        # 将QImage转换为numpy数组进行处理
        img = self.qimage_to_numpy(self.original_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用选定的效果
        if effect == "原始图片":
            processed_img = img
        elif effect == "灰度化":
            processed_img = grayscale_image(img)
        elif effect == "二值化":
            processed_img = binarize_image(img,127,0)
        elif effect == "反二值化":
            processed_img = binarize_image(img,127,1)
        elif effect == "均值滤波":
            processed_img = mean_filter(img)
        elif effect == "中值滤波":
            processed_img = median_filter(img)
        elif effect == "理想低通滤波":
            processed_img = ideal_low_pass_filter(img,cutoff=30)
        elif effect == "巴特沃斯低通滤波":
            processed_img = butterworth_low_pass_filter(img,cutoff=30,order=2)
        elif effect == "高斯低通滤波":
            processed_img = gaussian_low_pass_filter(img,cutoff=30)
        elif effect == "Roberts算子":
            processed_img = roberts_sharpen(img)
        elif effect == "Sobel算子":
            processed_img = sobel_sharpen(img)
        elif effect == "Prewitt算子":
            processed_img = prewitt_sharpen(img)
        elif effect == "Laplacian算子":
            processed_img = laplacian_sharpen(img)
        elif effect == "理想高通滤波":
            processed_img = ideal_high_pass_filter(img,cutoff=30)
        elif effect == "巴特沃斯高通滤波":
            processed_img = butterworth_high_pass_filter(img,cutoff=30,order=2)
        elif effect == "高斯高通滤波":
            processed_img = gaussian_high_pass_filter(img,cutoff=30)
        elif effect == "绘制直方图":
            processed_img = plot_histogram(img)
        elif effect == "对数变换":
            processed_img = log_transform(img)
        elif effect == "均衡化":
            processed_img = equalize_histogram(img)
        elif effect == "正规化":
            processed_img = histogram_normalization(img)
        elif effect == "图像缩放":
            processed_img = resize_image(img, 0.5, 0.5)
        elif effect == "图像旋转":
            processed_img = rotate_image(img, 45)
        elif effect == "图像平移":
            processed_img = translate_image(img, 100, 50)
        elif effect == "图像翻转":
            processed_img = flip_image(img, 1)
        elif effect == "仿射变换":
            src_tri = np.float32([[50, 50], [200, 50], [50, 200]])
            dst_tri = np.float32([[10, 100], [200, 50], [100, 250]])
            processed_img = affine_transform(img, src_tri, dst_tri)
        elif effect == "Roberts算子边缘检测":
            processed_img = roberts_edge(img)
        elif effect == "Prewitt算子边缘检测":
            processed_img = prewitt_edge(img)
        elif effect == "Sobel算子边缘检测":
            processed_img = sobel_edge(img)
        elif effect == "Laplacian算子边缘检测":
            processed_img = laplacian_edge(img)
        elif effect == "LoG边缘检测":
            processed_img = log_edge(img)
        elif effect == "Canny边缘检测":
            processed_img = canny_edge(img)
        elif effect == "霍夫变换直线检测":
            processed_img = hough_lines(img, use_probabilistic=True)
        elif effect == "1":
            processed_img = run_style_transfer(img,"../src/style_transfer/saved_models/candy.pth")
        elif effect == "2":
            processed_img = run_style_transfer(img,"../src/style_transfer/saved_models/mosaic.pth")
        elif effect == "3":
            processed_img = run_style_transfer(img,"../src/style_transfer/saved_models/rain_princess.pth")
        elif effect == "4":
            processed_img = run_style_transfer(img,"../src/style_transfer/saved_models/udnie.pth")
        else:
            processed_img = img

        # 将处理后的numpy数组转换回QImage
        self.processed_image = self.numpy_to_qimage(processed_img)

        # 显示处理后的图片
        pixmap = QPixmap.fromImage(self.processed_image)
        self.processed_label.setPixmap(
            pixmap.scaled(
                self.processed_label.width(),
                self.processed_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.status_label.setText(f"已应用效果: {effect}")

    def save_image(self):
        """保存处理后的图片"""
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "没有要保存的图片")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "",
            "PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;位图 (*.bmp)",
            options=options
        )

        if file_path:
            if self.processed_image.save(file_path):
                self.status_label.setText(f"图片已保存至: {file_path}")
                QMessageBox.information(self, "成功", "图片保存成功！")
            else:
                QMessageBox.warning(self, "错误", "无法保存图片")

    def qimage_to_numpy(self, qimage):
        """将QImage转换为numpy数组"""
        # 确保图片格式为RGB32
        if qimage.format() != QImage.Format_RGB32:
            qimage = qimage.convertToFormat(QImage.Format_RGB32)

        width = qimage.width()
        height = qimage.height()

        # 获取原始数据
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)

        # 创建numpy数组
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        # 转换为RGB（去除Alpha通道）
        return arr[:, :, :3]

    def numpy_to_qimage(self, np_array):
        """将numpy数组转换为QImage"""
        if np_array.dtype != np.uint8:
            np_array = np_array.astype(np.uint8)

        if np_array.ndim == 2:
            # 灰度图像
            height, width = np_array.shape
            bytes_per_line = width
            qimage = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return qimage.copy()

        elif np_array.ndim == 3:
            height, width, channels = np_array.shape
            if channels == 1:
                np_array = np.repeat(np_array, 3, axis=-1)
                channels = 3
            if channels == 4:
                np_array = np_array[:, :, :3]
            bytes_per_line = 3 * width
            qimage = QImage(np_array.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            return qimage.copy()

        raise ValueError("不支持的图像格式")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageConverterApp()
    window.show()
    sys.exit(app.exec_())