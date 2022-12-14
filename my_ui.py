import random
import sys
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui

import cv2
import numpy as np

import data.masked_aligned_dataset
import my_dialog

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util

sys.path.append(r"E:\LocalRepo\my-lpr-workspace\synthetic_generation")
from HR_KOR_LP_Generator import HR_KOR_LP_Generator
from HR_ParkingCloud_Loader import HR_ParkingCloud_Loader

class MyWindow(QDialog, my_dialog.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load_folder.clicked.connect(self.btn_load_folder_clicked)
        self.pushButton_play_pause.clicked.connect(self.btn_play_pause_clicked)
        self.pushButton_stop.clicked.connect(self.btn_stop_clicked)
        self.checkBox_random.clicked.connect(self.on_press_checkbox)

        self.generator = HR_KOR_LP_Generator(r'E:\LocalRepo\my-lpr-workspace\synthetic_generation\BetaType\korean_LP\P1-1')
        self.loader = None

        # 메인 타이머
        self.timer_main = QTimer(self)
        self.timer_main.setInterval(100)
        self.timer_main.timeout.connect(self.callback_timeout)
        self.idx = 0

        self.timer_singleshot = QTimer(self)
        self.timer_singleshot.singleShot(100, self.init_network)

    def init_network(self):
        self.opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        self.opt.num_threads = 0  # test code only supports num_threads = 0
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        self.model = create_model(self.opt)  # create a model given opt.model and other options
        self.model.setup(self.opt)  # regular setup: load and print networks; create schedulers
        if self.opt.eval:
            self.model.eval()

    def on_press_checkbox(self):
        if self.checkBox_random.isChecked():
            self.lineEdit_desired_label.setDisabled(True)
        else:
            self.lineEdit_desired_label.setEnabled(True)
            self.lineEdit_desired_label.setText('')

    def btn_load_folder_clicked(self):
        prefix_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if prefix_path != '':
            self.loader = HR_ParkingCloud_Loader(prefix_path)
            if self.loader.valid:
                QMessageBox.about(self, 'About', 'Loading success')
            else:
                QMessageBox.about(self, 'About', 'There is no jpg or xml')
                self.loader = None

    def btn_play_pause_clicked(self):
        if self.loader is None:
            QMessageBox.about(self, 'About', 'Load folder first')
            return

        if self.timer_main.isActive():
            self.timer_main.stop()
        else:
            self.timer_main.start()

    def btn_stop_clicked(self):
        self.timer_main.stop()
        self.idx = 0

    def callback_timeout(self):
        try:
            # P1만
            while True:
                plate_type, label, left, top, right, bottom = self.loader.parse_info(self.loader.list_xml[self.idx])
                if plate_type == 'P1':
                    break
                self.idx += 1

            # real image
            img_orig = self.loader.image_read(self.loader.list_jpg[self.idx])
            qsz = self.label_image_origin.size()
            img_orig_show = cv2.resize(img_orig, (qsz.width(), qsz.height()))
            h, w, c = img_orig_show.shape
            qImg = QtGui.QImage(img_orig_show.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            self.label_image_origin.setPixmap(pixmap)
            self.idx += 1

            # ref model
            if self.checkBox_random.isChecked():
                txt = '%02d' % random.randint(0,99)
                txt += random.choice(['가','나','다','라','마','거','너','더','러','머','버','서','어','저','고','노','도','로','모','보','소','오','조','구','두','무','수','누','루','부','우','주','하','허','호'])
                txt += '%04d' % random.randint(0,9999)
                self.lineEdit_desired_label.setText(txt)
            else:
                txt = self.lineEdit_desired_label.text()
                if txt == '' or len(txt) != 7:
                    txt = '52가3108'
            img_ref_model = self.generator.make_LP(txt)
            qsz = self.label_image_ref_model.size()
            img_ref_model_show = cv2.resize(img_ref_model, (qsz.width(), qsz.height()))
            h, w, c = img_ref_model_show.shape
            qImg = QtGui.QImage(img_ref_model_show.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            self.label_image_ref_model.setPixmap(pixmap)

            ######### image resizing
            if right-left > 360*1.5:
                df = 360/(right-left)
                if int(img_orig.shape[1]*df) < 720:  #리사이징 후 가로가 720이 넘는지
                    df = 720/img_orig.shape[1]
                if int(img_orig.shape[0]*df) < 720:  #리사이징 후 세로가 720이 넘는지
                    df = 720/img_orig.shape[0]
                left = int(left*df)
                top = int(top*df)
                right = int(right*df)
                bottom = int(bottom*df)
                img_orig = cv2.resize(img_orig,(0,0),fx=df,fy=df)

            ######################## superimpose image
            img_gen = self.generator.make_LP(label)
            img_gen = cv2.erode(img_gen, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            img_gen = cv2.resize(img_gen, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow('img_gen',img_gen)
            img_ref_model = cv2.erode(img_ref_model, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            img_ref_model = cv2.resize(img_ref_model, (0, 0), fx=0.5, fy=0.5)

            ### gen과 같은 크기로 4-corner를 crop & warp
            pt_affine_src = np.float32([[left, top], [right, top], [right, bottom]])
            pt_affine_dst = np.float32([[0, 0], [img_gen.shape[1], 0], img_gen.shape[1::-1]])
            A = cv2.getAffineTransform(pt_affine_src, pt_affine_dst)
            img_roi = cv2.warpAffine(img_orig, A, img_gen.shape[1::-1])
            # cv2.imshow('img_roi',img_roi)

            ### text area mask
            cr_x = int(self.generator.char_xywh[0][0] * 1) - 10
            cr_y = int(self.generator.char_xywh[0][1] * 1) - 10
            cr_w = (self.generator.char_xywh[1][0] * 6 + self.generator.char_xywh[5][0]) * 1 + 20
            cr_h = self.generator.char_xywh[1][1] * 1 + 20
            mask_text_area = np.zeros_like(img_gen[:, :, 0])
            mask_text_area[cr_y:cr_y + cr_h, cr_x:cr_x + cr_w] = 255
            ### feature extraction
            img_gen_gray = cv2.cvtColor(img_gen, cv2.COLOR_BGRA2GRAY)
            pt_gen = cv2.goodFeaturesToTrack(img_gen_gray, 500, 0.01, 5, mask=mask_text_area)
            ### optional : feature extraction visualization
            # img_gen_cornerdisp = cv2.cvtColor(img_gen_gray,cv2.COLOR_GRAY2BGR)
            # for pt in pt_gen:
            #     cv2.circle(img_gen_cornerdisp, tuple(map(int, pt[0])), 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow('img_gen_cornerdisp', img_gen_cornerdisp)

            # img_gencrop2 = np.zeros_like(img_gen2[:,:,:3])
            # img_gencrop2[cr_y2:cr_y2+cr_h2,cr_x2:cr_x2+cr_w2,:]=img_gen2[cr_y2:cr_y2+cr_h2,cr_x2:cr_x2+cr_w2,:3].copy()

            ### histogram equalization
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            img_roi_gray_histeq = cv2.equalizeHist(img_roi_gray)
            # cv2.imshow('img_roi_gray_histeq',img_roi_gray_histeq)
            ### feature tracking
            pt_tracked, status, err = cv2.calcOpticalFlowPyrLK(img_gen_gray, img_roi_gray_histeq, pt_gen, None)
            ### optional : feature tracking visualization
            # img_blend_disp = cv2.addWeighted(img_gen[:,:,:3], 0.5, img_roi, 0.5, 0)
            # for i in range(len(pt_tracked)):
            #     if status[i,0] == 0: # status = 0인 것은 제외, 잘못 찾은 것을 의미
            #         continue
            #     cv2.line(img_blend_disp, tuple(map(int, pt_gen[i, 0])), tuple(map(int, pt_tracked[i, 0])), (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.circle(img_blend_disp, tuple(map(int, pt_gen[i, 0])), 3, (0, 0, 255), 2, cv2.LINE_AA)
            #     cv2.circle(img_blend_disp, tuple(map(int, pt_tracked[i, 0])), 3, (0, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('img_blend_disp', img_blend_disp)

            ### find homography
            img_roi_th_orig = cv2.adaptiveThreshold(img_roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                    31, 0)
            # img_roi_th_orig = cv2.adaptiveThreshold(img_roi_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,71,0)
            # _,img_roi_th_orig = cv2.threshold(img_roi_gray_histeq,0,255,cv2.THRESH_OTSU)
            pt1 = pt_gen[status == 1].astype(np.int32)
            pt2 = pt_tracked[status == 1].astype(np.int32)
            minVal = 9999999999
            minIdx = -1
            list_img_warped = []
            if len(pt1) >= 4:
                for ransac_th in range(1, 15):
                    H, stat = cv2.findHomography(pt1, pt2, cv2.RANSAC, ransac_th)
                    if H is not None:
                        img_warped = cv2.warpPerspective(img_gen[:, :, :3], H, img_gen.shape[1::-1])
                        list_img_warped.append(img_warped)

                        img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                        # img_warped_th = cv2.adaptiveThreshold(img_warped_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,0)
                        # img_warped_th = cv2.adaptiveThreshold(img_warped_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,0)
                        _, img_warped_th = cv2.threshold(img_warped_gray, 0, 255, cv2.THRESH_OTSU)

                        mask_warped = cv2.warpPerspective(mask_text_area, H, img_gen.shape[1::-1])
                        img_roi_th = cv2.bitwise_and(img_roi_th_orig, mask_warped)
                        img_warped_th = cv2.bitwise_and(img_warped_th, mask_warped)

                        # optional : in,outlier visualization
                        # img_inoutlier = img_warped.copy()
                        # for i,good in enumerate(stat):
                        #     color = (0,255,0) if good else (0,0,255)
                        #     cv2.line(img_inoutlier, pt1[i,:], pt2[i,:], color, 2, cv2.LINE_AA)

                        img_subtract = cv2.absdiff(img_warped_th, img_roi_th)
                        s = img_subtract.sum() / mask_warped.sum()
                        if minVal > s:
                            minVal = s
                            minIdx = ransac_th - 1
                            # cv2.imshow('img_warped',img_warped)
                            # cv2.imshow('mask_warped',mask_warped)
                            # cv2.imshow('img_warped_th',img_warped_th)
                            # cv2.imshow('img_roi_th',img_roi_th)
                            # cv2.imshow('img_subtract',img_subtract)
                            # cv2.imshow('img_inoutlier',img_inoutlier)
                            # print(H)
                            HH = np.reshape(H, 9)

            D = HH[0] * HH[4] - HH[1] * HH[3]
            sx = (HH[0] ** 2 + HH[3] ** 2) ** 0.5
            sy = (HH[1] ** 2 + HH[4] ** 2) ** 0.5
            P = (HH[6] ** 2 + HH[7] ** 2) ** 0.5
            if D <= 0:
                print('D<=0')
            if sx < 0.1:
                print('sx<0.1')
            if sx > 4:
                print('sx>4')
            if sy < 0.1:
                print('sy<0.1')
            if sy > 4:
                print('sy>4')
            if P > 0.001:
                print('P>0.001')

            # a=0
            # # while True:
            # for _ in range(6):
            #     key_in = cv2.waitKey(200)
            #     if key_in != -1 or minIdx == -1:
            #         break
            #     else:
            #         cv2.imshow('img_warped',list_img_warped[minIdx] if a==0 else img_roi)
            #     a=1-a

            A_ = np.append(A, [[0, 0, 1]], axis=0)
            A_inv = np.linalg.inv(A_)
            H_ = HH.reshape(3, 3)
            T = A_inv.dot(H_)

            # img_gen_warped_big = cv2.warpPerspective(img_gen[:,:,:3], T, img_orig.shape[1::-1])
            # cv2.imshow('img_gen_warped_big',img_gen_warped_big)

            # text area crop
            cr_x0 = cr_x + 10
            cr_y0 = cr_y + 10
            cr_w0 = cr_w - 20
            cr_h0 = cr_h - 20

            ###### 원본과 같은 번호판을 겹칠 때
            # img_gencrop = np.zeros_like(img_gen[:, :, :3])
            # img_gencrop[cr_y0:cr_y0 + cr_h0, cr_x0:cr_x0 + cr_w0, :] = img_gen[cr_y0:cr_y0 + cr_h0, cr_x0:cr_x0 + cr_w0,:3].copy()
            ###### 원본과 다른 번호판을 겹칠 때
            img_gencrop = np.zeros_like(img_ref_model[:, :, :3])
            img_gencrop[cr_y0:cr_y0 + cr_h0, cr_x0:cr_x0 + cr_w0, :] = img_ref_model[cr_y0:cr_y0 + cr_h0, cr_x0:cr_x0 + cr_w0,:3].copy()

            img_gencrop_warped_big = cv2.warpPerspective(img_gencrop, T, img_orig.shape[1::-1])

            sq_center_x = (left + right) // 2
            sq_center_y = (top + bottom) // 2
            sq_lr = np.array([sq_center_x - 360, sq_center_x + 360])
            sq_tb = np.array([sq_center_y - 360, sq_center_y + 360])
            if sq_lr[0] < 0:
                sq_lr -= sq_lr[0]
            if img_orig.shape[1] - sq_lr[1] <= 0:
                sq_lr += (img_orig.shape[1] - sq_lr[1])
            if sq_tb[0] < 0:
                sq_tb -= sq_tb[0]
            if img_orig.shape[0] - sq_tb[1] <= 0:
                sq_tb += (img_orig.shape[0] - sq_tb[1])

            # if pt_lt[0] < 0:
            #     pt_rb[0] -= pt_lt[0]
            #     pt_lt[0] = 0
            # if pt

            mask_text_area2 = np.zeros_like(img_gen[:, :, 0])
            mask_text_area2[cr_y0:cr_y0 + cr_h0, cr_x0:cr_x0 + cr_w0] = 255
            img_mask_warped_big = cv2.warpPerspective(mask_text_area2, T, img_orig.shape[1::-1])
            # _, mask = cv2.threshold(img_gencrop_warped_big[:,:,0],0,255,cv2.THRESH_BINARY) # 투명도가 0이 아닌 모든 부분 추출
            mask = img_mask_warped_big
            img1 = cv2.bitwise_and(img_orig, img_orig, mask=cv2.bitwise_not(mask))  # 안 겹치는 부분은 img1에 bg를 그대로 넣음
            img2 = cv2.bitwise_and(img_gencrop_warped_big, img_gencrop_warped_big, mask=mask)  # 겹치는 부분은 mask에 따라 우선 추출
            img_overlay = img1 + img2  # 두 개는 서로 배타적이라 그냥 더해도 overflow 나지 않음

            img_orig_squarecrop = img_orig[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], :]
            img_overlay_sqaurecrop = img_overlay[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], :]
            img_mask_squarecrop = cv2.cvtColor(mask[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1]], cv2.COLOR_GRAY2BGR)

            ABM = np.hstack([img_overlay_sqaurecrop, img_orig_squarecrop, img_mask_squarecrop])

            qsz = self.label_image_superimpose.size()
            ABM2 = cv2.resize(ABM, (qsz.width(), qsz.height()))
            h, w, c = ABM2.shape
            qImg = QtGui.QImage(ABM2.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            self.label_image_superimpose.setPixmap(pixmap)

            ##### pytorch 연산 부분
            tensor_data = data.masked_aligned_dataset.MaskedAlignedDataset.my_getitem(self.opt, ABM)
            self.model.set_input(tensor_data)
            self.model.test()
            im_data = getattr(self.model, 'fake_B') # visuals = self.model.get_current_visuals()
            im = util.tensor2im(im_data)

            ### 다시 원본과 결합
            im_rsz = cv2.resize(im, (720,720))
            img_orig[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], :] = im_rsz.copy()

            qsz = self.label_image_gen.size()
            im2 = cv2.resize(img_orig, (qsz.width(), qsz.height()))
            h, w, c = im2.shape
            qImg = QtGui.QImage(im2.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            self.label_image_gen.setPixmap(pixmap)

        except Exception as e:
            print(e)
            print(self.idx)
            self.timer_main.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()