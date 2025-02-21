# -*- coding:utf-8 -*-
import PySide2
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtWidgets, QtCorels
import sys
from PySide2.QtGui import QIcon
import win1, win2, win3, instru

class main:

    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        self.ui = QUiLoader().load('ui/main.ui')   #控件都变成了ui的属性

        self.ui.bt1.clicked.connect(self.handle1)
        self.ui.bt2.clicked.connect(self.handle2)
        self.ui.bt3.clicked.connect(self.handle3)
        self.ui.bt4.clicked.connect(self.handle4)
        self.ui.instru.clicked.connect(self.handle)

    def handle1(self):
        self.ui.window2 = win1.win1()
        # 显示新窗口
        self.ui.window2.ui.setWindowIcon(QIcon('logo.png'))
        self.ui.window2.ui.show()
        # 关闭自己
        self.ui.close()


    def handle2(self):
        self.ui.window2 = win2.win2()
        # 显示新窗口
        self.ui.window2.ui.setWindowIcon(QIcon('logo.png'))
        self.ui.window2.ui.show()
        # 关闭自己
        self.ui.close()

    def handle3(self):
        self.ui.window2 = win3.win3()
        # 显示新窗口
        self.ui.window2.ui.setWindowIcon(QIcon('logo.png'))
        self.ui.window2.ui.show()
        # 关闭自己
        self.ui.close()

    def handle4(self):
        # 关闭自己
        self.ui.close()


    def handle(self):
        self.ui.window2 = instru.instru()
        # 显示新窗口
        self.ui.window2.ui.setWindowIcon(QIcon('logo.png'))
        self.ui.window2.ui.show()
        # 关闭自己
        self.ui.close()


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    # app.setWindowIcon(QIcon('logo.png'))
    window = main()
    window.ui.setWindowIcon(QIcon('logo.png'))
    window.ui.show()
    sys.exit(app.exec_())
