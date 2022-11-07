# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'my_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(975, 770)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_left = QtWidgets.QVBoxLayout()
        self.verticalLayout_left.setObjectName("verticalLayout_left")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_left.addItem(spacerItem)
        self.pushButton_load_folder = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_load_folder.sizePolicy().hasHeightForWidth())
        self.pushButton_load_folder.setSizePolicy(sizePolicy)
        self.pushButton_load_folder.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_load_folder.setObjectName("pushButton_load_folder")
        self.verticalLayout_left.addWidget(self.pushButton_load_folder)
        self.horizontalLayout_upper = QtWidgets.QHBoxLayout()
        self.horizontalLayout_upper.setObjectName("horizontalLayout_upper")
        self.pushButton_play_pause = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_play_pause.sizePolicy().hasHeightForWidth())
        self.pushButton_play_pause.setSizePolicy(sizePolicy)
        self.pushButton_play_pause.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_play_pause.setObjectName("pushButton_play_pause")
        self.horizontalLayout_upper.addWidget(self.pushButton_play_pause)
        self.pushButton_stop = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_stop.sizePolicy().hasHeightForWidth())
        self.pushButton_stop.setSizePolicy(sizePolicy)
        self.pushButton_stop.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.horizontalLayout_upper.addWidget(self.pushButton_stop)
        self.verticalLayout_left.addLayout(self.horizontalLayout_upper)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_left.addItem(spacerItem1)
        self.horizontalLayout_lower = QtWidgets.QHBoxLayout()
        self.horizontalLayout_lower.setObjectName("horizontalLayout_lower")
        self.lineEdit_desired_label = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_desired_label.setText("")
        self.lineEdit_desired_label.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_desired_label.setObjectName("lineEdit_desired_label")
        self.horizontalLayout_lower.addWidget(self.lineEdit_desired_label)
        self.checkBox_random = QtWidgets.QCheckBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_random.sizePolicy().hasHeightForWidth())
        self.checkBox_random.setSizePolicy(sizePolicy)
        self.checkBox_random.setObjectName("checkBox_random")
        self.horizontalLayout_lower.addWidget(self.checkBox_random)
        self.verticalLayout_left.addLayout(self.horizontalLayout_lower)
        self.groupBox_ref_model = QtWidgets.QGroupBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_ref_model.sizePolicy().hasHeightForWidth())
        self.groupBox_ref_model.setSizePolicy(sizePolicy)
        self.groupBox_ref_model.setObjectName("groupBox_ref_model")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_ref_model)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_3.addItem(spacerItem2)
        self.label_image_ref_model = QtWidgets.QLabel(self.groupBox_ref_model)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image_ref_model.sizePolicy().hasHeightForWidth())
        self.label_image_ref_model.setSizePolicy(sizePolicy)
        self.label_image_ref_model.setMinimumSize(QtCore.QSize(0, 100))
        self.label_image_ref_model.setText("")
        self.label_image_ref_model.setObjectName("label_image_ref_model")
        self.verticalLayout_3.addWidget(self.label_image_ref_model)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_3.addItem(spacerItem3)
        self.verticalLayout_left.addWidget(self.groupBox_ref_model)
        self.groupBox_superimpose = QtWidgets.QGroupBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_superimpose.sizePolicy().hasHeightForWidth())
        self.groupBox_superimpose.setSizePolicy(sizePolicy)
        self.groupBox_superimpose.setObjectName("groupBox_superimpose")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_superimpose)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_image_superimpose = QtWidgets.QLabel(self.groupBox_superimpose)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image_superimpose.sizePolicy().hasHeightForWidth())
        self.label_image_superimpose.setSizePolicy(sizePolicy)
        self.label_image_superimpose.setMinimumSize(QtCore.QSize(0, 160))
        self.label_image_superimpose.setText("")
        self.label_image_superimpose.setObjectName("label_image_superimpose")
        self.verticalLayout.addWidget(self.label_image_superimpose)
        self.verticalLayout_left.addWidget(self.groupBox_superimpose)
        self.horizontalLayout.addLayout(self.verticalLayout_left)
        self.verticalLayout_right = QtWidgets.QVBoxLayout()
        self.verticalLayout_right.setObjectName("verticalLayout_right")
        self.groupBox_origin = QtWidgets.QGroupBox(Dialog)
        self.groupBox_origin.setObjectName("groupBox_origin")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_origin)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_image_origin = QtWidgets.QLabel(self.groupBox_origin)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image_origin.sizePolicy().hasHeightForWidth())
        self.label_image_origin.setSizePolicy(sizePolicy)
        self.label_image_origin.setText("")
        self.label_image_origin.setObjectName("label_image_origin")
        self.horizontalLayout_7.addWidget(self.label_image_origin)
        self.verticalLayout_right.addWidget(self.groupBox_origin)
        self.groupBox_gen = QtWidgets.QGroupBox(Dialog)
        self.groupBox_gen.setObjectName("groupBox_gen")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_gen)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_image_gen = QtWidgets.QLabel(self.groupBox_gen)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image_gen.sizePolicy().hasHeightForWidth())
        self.label_image_gen.setSizePolicy(sizePolicy)
        self.label_image_gen.setText("")
        self.label_image_gen.setObjectName("label_image_gen")
        self.horizontalLayout_6.addWidget(self.label_image_gen)
        self.verticalLayout_right.addWidget(self.groupBox_gen)
        self.horizontalLayout.addLayout(self.verticalLayout_right)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Synthetic pseudo-real License Plate Generation DEMO"))
        self.pushButton_load_folder.setText(_translate("Dialog", "Load Data"))
        self.pushButton_play_pause.setText(_translate("Dialog", "Play / Pause"))
        self.pushButton_stop.setText(_translate("Dialog", "Stop"))
        self.lineEdit_desired_label.setPlaceholderText(_translate("Dialog", "12가3456"))
        self.checkBox_random.setText(_translate("Dialog", "RANDOM?"))
        self.groupBox_ref_model.setTitle(_translate("Dialog", "Synthetic Reference Model Generation"))
        self.groupBox_superimpose.setTitle(_translate("Dialog", "Superimposed Image (input)"))
        self.groupBox_origin.setTitle(_translate("Dialog", "Real Image"))
        self.groupBox_gen.setTitle(_translate("Dialog", "Synthetic Pseudo-real Image (output)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
