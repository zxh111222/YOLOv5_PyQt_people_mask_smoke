import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import identify


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        # ========= 参数定义 ===========
        self.input_image = None     # 输入图像
        self.output_image = None    # 输出图像
        self.identify_labels = []   # 检测结果
        self.identify_api = identify.Identify()  # 调用识别API
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.show_video)  # 定时器执行show_video函数
        # ========= 界面生成 ===========
        super(MainUi, self).__init__()
        self.resize(860, 590)
        self.setWindowTitle('基于YOLOv5的人流计数系统')  # 设置窗口标题
        # 导航文字区
        self.label = QLabel(self)
        self.label.setText("基于YOLOv5的人流计数系统")
        self.label.setFixedSize(860, 40)
        self.label.move(0, 0)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 水平垂直居中
        self.label.setStyleSheet("QLabel{font-size:20px;font-weight:bold;font-family:'黑体'; }")
        # 原图显示区域
        self.input_img = QLabel(self)
        self.input_img.setFixedSize(400, 400)
        self.input_img.move(20, 50)
        self.input_img.setAlignment(Qt.AlignCenter)
        self.input_img.setStyleSheet("QLabel{border: 2px solid gray;}")
        # 检测图显示区域
        self.output_img = QLabel(self)
        self.output_img.setFixedSize(400, 400)
        self.output_img.move(440, 50)
        self.output_img.setAlignment(Qt.AlignCenter)
        self.output_img.setStyleSheet("QLabel{border: 2px solid gray;}")
        # 检测结果显示区域
        self.result = QLabel(self)
        self.result.setText("检测人数：")
        self.result.setFixedSize(820, 40)
        self.result.move(20, 470)
        self.result.setStyleSheet("QLabel{border: 2px solid gray;font-size:20px;font-family:'黑体'; }")
        # 图片检测按钮
        self.function1 = QPushButton(self)
        self.function1.setText("图像识别")
        self.function1.resize(260, 40)
        self.function1.move(20, 530)
        self.function1.setCursor(Qt.PointingHandCursor)
        self.function1.setStyleSheet("QPushButton{border: 2px solid gray;font-size:20px;font-family:'黑体'; }")
        self.function1.clicked.connect(self.show_image)
        # 视频检测按钮
        self.function2 = QPushButton(self)
        self.function2.setText("开启视频识别")
        self.function2.resize(260, 40)
        self.function2.move(300, 530)
        self.function2.setCursor(Qt.PointingHandCursor)
        self.function2.setStyleSheet("QPushButton{border: 2px solid gray;font-size:20px;font-family:'黑体'; }")
        self.function2.clicked.connect(self.video_identify)
        # 摄像头检测按钮
        self.function3 = QPushButton(self)
        self.function3.setText("开启摄像头识别")
        self.function3.resize(260, 40)
        self.function3.move(580, 530)
        self.function3.setCursor(Qt.PointingHandCursor)
        self.function3.setStyleSheet("QPushButton{border: 2px solid gray;font-size:20px;font-family:'黑体'; }")
        self.function3.clicked.connect(self.camera_identify)

    # 图片检测
    def show_image(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "./", "*.jpg;*.png;;All Files(*)")
        if len(image_path) < 5:
            self.reset()
        self.input_image = cv2.imread(image_path)
        self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(self.input_image, False)
        # 图像二次处理
        self.input_image1, self.output_image1, self.identify_labels = self.identify_api.show_frame(self.input_image, False)
        
        if self.output_image is not None:
            self.input_image = self.change_image(self.input_image)
            self.output_image = self.change_image(self.output_image)
            # 将检测图像画面显示在界面
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(self.input_image.data, self.input_image.shape[1], self.input_image.shape[0],
                                    self.input_image.shape[1] * 3, QImage.Format_RGB888)
            self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
            self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2RGB)
            show_output_img = QImage(self.output_image.data, self.output_image.shape[1], self.output_image.shape[0],
                                     self.output_image.shape[1] * 3, QImage.Format_RGB888)
            self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
            # 将结果显示在界面
            self.result.setText("检测人数：" + str(len(self.identify_labels)))
        else:
            self.reset()

    # 视频检测
    def video_identify(self):
        if self.function2.text() == "开启视频识别" and not self.timer_video.isActive():
            video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "打开视频", "", "*.mp4;*.avi;;All Files(*)")
            if len(video_path) < 5:
                self.reset()
            flag = self.identify_api.cap.open(video_path)
            if flag is False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_video.start(30)
                self.function1.setDisabled(True)
                self.function3.setDisabled(True)
                self.function2.setText("关闭视频识别")
        else:
            self.identify_api.cap.release()
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function3.setDisabled(False)
            self.function2.setText("开启视频识别")
            self.reset()

    # 摄像头检测
    def camera_identify(self):
        if self.function3.text() == "开启摄像头识别" and not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.identify_api.cap.open(0)
            if flag is False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_video.start(30)
                self.function1.setDisabled(True)
                self.function2.setDisabled(True)
                self.function3.setText("关闭摄像头识别")
        else:
            self.identify_api.cap.release()
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function2.setDisabled(False)
            self.function3.setText("开启摄像头识别")
            self.reset()

    # 展示图像与显示结果(视频与摄像头)
    def show_video(self):
        self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(None, True)
        if self.output_image is not None:
            self.input_image = self.change_image(self.input_image)
            self.output_image = self.change_image(self.output_image)
            # 将检测图像画面显示在界面
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(self.input_image.data, self.input_image.shape[1], self.input_image.shape[0],
                                    self.input_image.shape[1] * 3, QImage.Format_RGB888)
            self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
            self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2RGB)
            show_output_img = QImage(self.output_image.data, self.output_image.shape[1], self.output_image.shape[0],
                                     self.output_image.shape[1] * 3, QImage.Format_RGB888)
            self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
            # 将结果显示在界面
            self.result.setText("检测人数：" + str(len(self.identify_labels)))
        else:
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function2.setDisabled(False)
            self.function3.setDisabled(False)
            self.function2.setText("开启视频识别")
            self.function3.setText("开启摄像头识别")
            self.reset()

    # 改变图像大小在界面显示
    @staticmethod
    def change_image(input_image):
        if input_image is not None:
            # 更换为界面适应性大小显示
            wh = float(int(input_image.shape[0]) / int(input_image.shape[1]))
            show_wh = 1
            if int(input_image.shape[0]) > 400 or int(input_image.shape[1]) > 400:
                if show_wh - wh < 0:
                    h = 400
                    w = int(h / wh)
                    output_image = cv2.resize(input_image, (w, h))
                else:
                    w = 400
                    h = int(w * wh)
                    output_image = cv2.resize(input_image, (w, h))
            else:
                output_image = input_image
            return output_image
        else:
            return input_image

    # 清空数据
    def reset(self):
        self.input_image = None  # 输入图像
        self.output_image = None  # 输出图像
        self.identify_labels = []  # 检测结果
        self.input_img.clear()   # 清空输入图像显示区
        self.output_img.clear()  # 清空输出图像显示区
        self.result.setText("检测人数：")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainUi = MainUi()
    mainUi.show()
    sys.exit(app.exec_())