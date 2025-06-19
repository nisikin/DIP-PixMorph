import os
import sys

from PyQt5.QtCore import Qt,QThread, pyqtSignal
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
    QDesktopWidget,
    QLineEdit,
)
# 导入图像处理模块
from src.basic.binarize import *
from src.basic.edge_detection import *
from src.basic.filtering import *
from src.basic.geometry import *
from src.basic.grayscale import *
from src.basic.histogram import *
from src.basic.salt_pepper_noise import *
from src.basic.morph_ops import *

from src.style_transfer.neural_style.run import *
from src.pixel.convert import *

class ImageConverterApp(QMainWindow):

    """图片转换工具的主窗口类，基于PyQt5实现图像处理GUI"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.original_image = None  # 原始图片
        self.processed_image = None  # 处理后的图片
        self.initUI()
        self.setWindowTitle("图片转换工具")  # 设置窗口标题
        screen = QDesktopWidget().availableGeometry()
        width, height = 1600, 900
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.setGeometry(x, y, width, height)

    def initUI(self, control_layout=None):
        """初始化用户界面"""
        # 定义效果选项字典，包含各种图像处理类别及其子选项
        self.effect_options = {
            "原始图片":["原始图片"],
            "灰度化": ["灰度化转换"],
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
            "几何变换": ["图像缩放", "图像旋转", "图像平移", "图像垂直翻转","图像水平翻转","图像垂直水平翻转", "仿射变换"],
            "边缘检测": [
                "Roberts算子边缘检测",
                "Prewitt算子边缘检测",
                "Sobel算子边缘检测",
                "Laplacian算子边缘检测",
                "LoG边缘检测",
                "Canny边缘检测",
                "霍夫变换直线检测",
            ],
            "椒盐噪声":["添加椒盐噪声"],
            "图像形态学操作":["腐蚀","膨胀","开运算","闭运算"],
            "风格迁移": ["糖果", "马赛克", "雨中公主", "Udine","test"],
            "像素凤转换":["pixel"],
        }

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 顶部控制区域（水平布局）
        main_layout = QVBoxLayout()

        # 顶部控制区域
        param_layout = QVBoxLayout()
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

        # 输入框
        self.param1_input = QLineEdit()
        self.param1_input.setPlaceholderText("参数1")
        self.param1_input.setFixedSize(40, 25)

        # 输入框2
        self.param2_input = QLineEdit()
        self.param2_input.setPlaceholderText("参数2")
        self.param2_input.setFixedSize(40, 25)

        param_layout.addWidget(self.param1_input)
        param_layout.addWidget(self.param2_input)

        # 将参数输入框加入顶部控制区域
        control_layout.addLayout(param_layout)

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
        original_group.setStyleSheet("QGroupBox { font-size: 18px; }")
        original_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(800, 800)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        original_layout.addWidget(self.original_label)

        # 分辨率标签（原图）
        self.original_resolution_label = QLabel("分辨率：- × -")
        self.original_resolution_label.setAlignment(Qt.AlignLeft)
        self.original_resolution_label.setFixedHeight(15)
        original_layout.addWidget(self.original_resolution_label)

        original_group.setLayout(original_layout)
        image_layout.addWidget(original_group)

        # 处理后的图片显示
        processed_group = QGroupBox("转换结果")
        processed_group.setStyleSheet("QGroupBox { font-size: 18px; }")
        processed_layout = QVBoxLayout()
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(800, 800)
        self.processed_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        processed_layout.addWidget(self.processed_label)

        # 分辨率标签（处理后）
        self.processed_resolution_label = QLabel("分辨率：- × -")
        self.processed_resolution_label.setAlignment(Qt.AlignLeft)
        self.processed_resolution_label.setFixedHeight(15)
        processed_layout.addWidget(self.processed_resolution_label)

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
                self.original_label.setPixmap(pixmap)
                self.original_resolution_label.setText(f"分辨率：{pixmap.width()} × {pixmap.height()}")
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件")

    def process_image(self):
        """应用选定的图片处理效果"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return

        # 获取选定的效果
        effect = self.detail_combo.currentText()
        val1 = self.get_input_params1()
        val2 = self.get_input_params2()

        # 将QImage转换为numpy数组进行处理
        img = self.qimage_to_numpy(self.original_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用选定的效果
        if effect == "原始图片":
            processed_img = img
        elif effect == "灰度化转换":
            processed_img = grayscale_image(img)
        elif effect == "二值化":
            if val1 is None :
                processed_img = binarize_image(img,127, 0)
            else:
                processed_img = binarize_image(img,val1,0)
        elif effect == "反二值化":
            if val1 is None:
                processed_img = binarize_image(img,127,1)
            else:
                processed_img = binarize_image(img, val1, 1)
        elif effect == "均值滤波":
            processed_img = mean_filter(img)
        elif effect == "中值滤波":
            processed_img = median_filter(img)
        elif effect == "理想低通滤波":
            if val1 is None:
                processed_img = ideal_low_pass_filter(img,cutoff=30)
            else:
                processed_img = ideal_low_pass_filter(img,val1)
        elif effect == "巴特沃斯低通滤波":
            if val1 is None or val2 is None:
                processed_img = butterworth_low_pass_filter(img,cutoff=30,order=2)
            else:
                processed_img = butterworth_low_pass_filter(img,val1,val2)
        elif effect == "高斯低通滤波":
            if val1 is None:
                processed_img = gaussian_low_pass_filter(img,cutoff=30)
            else:
                processed_img = gaussian_low_pass_filter(img,val1)
        elif effect == "Roberts算子":
            processed_img = roberts_sharpen(img)
        elif effect == "Sobel算子":
            processed_img = sobel_sharpen(img)
        elif effect == "Prewitt算子":
            processed_img = prewitt_sharpen(img)
        elif effect == "Laplacian算子":
            processed_img = laplacian_sharpen(img)
        elif effect == "理想高通滤波":
            if val1 is None:
                processed_img = ideal_high_pass_filter(img,cutoff=30)
            else:
                processed_img = ideal_high_pass_filter(img,val1)
        elif effect == "巴特沃斯高通滤波":
            if val1 is None or val2 is None:
                processed_img = butterworth_high_pass_filter(img,cutoff=30,order=2)
            else:
                processed_img = butterworth_high_pass_filter(img,val1,val2)
        elif effect == "高斯高通滤波":
            if val1 is None:
                processed_img = gaussian_high_pass_filter(img,cutoff=30)
            else:
                processed_img = gaussian_high_pass_filter(img,val1)
        elif effect == "绘制直方图":
            processed_img = plot_histogram(img)
        elif effect == "对数变换":
            processed_img = log_transform(img)
        elif effect == "均衡化":
            processed_img = equalize_histogram(img)
        elif effect == "正规化":
            processed_img = histogram_normalization(img)
        elif effect == "图像缩放":
            if val1 is None or val2 is None:
                processed_img = resize_image(img, 0.5, 0.5)
            else:
                processed_img = resize_image(img,val1,val2)
        elif effect == "图像旋转":
            if val1 is None:
                processed_img = rotate_image(img, 45)
            else:
                processed_img = rotate_image(img, val1)
        elif effect == "图像平移":
            if val1 is None or val2 is None:
                processed_img = translate_image(img, 100, 50)
            else:
                processed_img = translate_image(img,val1,val2)
        elif effect == "图像垂直翻转":
            processed_img = flip_image(img, 0)
        elif effect == "图像水平翻转":
            processed_img = flip_image(img, 1)
        elif effect == "图像垂直水平翻转":
            processed_img = flip_image(img, -1)
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
        elif effect == "添加椒盐噪声":
            if val1 is None:
                processed_img = add_salt_pepper_noise(img)
            else:
                processed_img = add_salt_pepper_noise(img,val1)
        elif effect == "腐蚀":
            processed_img = erode_image(img)
        elif effect == "膨胀":
            processed_img = dilate_image(img)
        elif effect == "开运算":
            processed_img = open_image(img)
        elif effect == "闭运算":
            processed_img = close_image(img)
        elif effect == "糖果":
            processed_img = run_style_transfer(img, "../src/style_transfer/models/candy.pth")
        elif effect == "马赛克":
            processed_img = run_style_transfer(img, "../src/style_transfer/models/mosaic.pth")
        elif effect == "雨中公主":
            processed_img = run_style_transfer(img, "../src/style_transfer/models/rain_princess.pth")
        elif effect == "Udine":
            processed_img = run_style_transfer(img, "../src/style_transfer/models/udnie.pth")
        elif effect == "test":
            processed_img = run_style_transfer(img,"../src/style_transfer/models/epoch_10_Sun_Jun_15_17:07:18_2025_100000.0_10000000000.0.model")
        elif effect == "pixel":
            processed_img = convert_to_pixel(img)
        else:
            processed_img = img

        # 将处理后的numpy数组转换回QImage
        self.processed_image = self.numpy_to_qimage(np.array(processed_img))

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

        self.processed_resolution_label.setText(
            f"分辨率：{pixmap.width()} × {pixmap.height()}"
        )

        self.status_label.setText(f"已应用效果: {effect}")

    def get_input_params1(self):
        """
        获取输入框的参数
        """
        try:
            val1 = float(self.param1_input.text())
            return val1
        except ValueError:
            return None

    def get_input_params2(self):
        """
            获取输入框的参数
        """
        try:
            val2 = float(self.param2_input.text())
            return val2
        except ValueError:
            return None

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
        ptr.setsize(height * width * 4)  # 4 bytes per photo2pixel (RGBA)

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