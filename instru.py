from PySide2 import QtCore
from PySide2.QtWidgets import QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
import PySide2
from PySide2.QtGui import QIcon
import main
import function

class instru:

    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        self.ui = QUiLoader().load('ui/instru.ui')   #控件都变成了ui的属性

        self.ui.exit.clicked.connect(self.process_exit)  # 退出按钮事件


    def process_exit(self):
        self.ui.close()
        self.ui.window = main.main()
        self.ui.window.ui.setWindowIcon(QIcon('logo.png'))
        self.ui.window.ui.show()

