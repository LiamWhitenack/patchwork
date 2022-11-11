import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import Qt
import numpy as np
from patchwork_ui import Ui_MainWindow
import struct
import pandas as pd
from random import shuffle

# self.MoveAheadButton_p2.setText(_translate("MainWindow", "Move: 1 Space(s)\nReceive: 1 Button(s)"))

WHITE = (255,255,255)
BLACK = (0,0,0)

def rotate(l, n):
    return l[n:] + l[:n]

def hex_to_rgb(hex:str):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)
  
    return tuple(rgb)

'''This terrible block of code makes a dictionary mapping each space to a point on the time board. Feel free to ignore it'''

time_space = {}
for i in range(9):
    time_space[i] = {'space_coor':(0, 8 - i)} # type: ignore
for j in range(5):
    time_space[i + j + 1] = {'space_coor':(j + 1, 0)} # type: ignore
for k in range(8):
    time_space[i + j + k + 2] = {'space_coor':(5, k + 1)} # type: ignore
for l in range(4):
    time_space[i + j + k + l + 3] = {'space_coor':(4 - l, k + 1)} # type: ignore
for m in range(7):
    time_space[i + j + k + l + m + 4] = {'space_coor':(1, 7 - m)} # type: ignore
for n in range(3):
    time_space[i + j + k + l + m + n + 5] = {'space_coor':(2 + n, 1)} # type: ignore
for o in range(6):
    time_space[i + j + k + l + m + n + o + 6] = {'space_coor':(4, 2 + o)} # type: ignore
time_space[i + j + k + l + m + n + o + 7] = {'space_coor':(3, 7)} # type: ignore
for p in range(6):
    time_space[i + j + k + l + m + n + o + p + 8] = {'space_coor':(2, 7 - p)} # type: ignore
for q in range(5):
    time_space[i + j + k + l + m + n + o + p + q + 9] = {'space_coor':(3, 2 + q)} # type: ignore

# it's not stricly necessary to subclass Ui_MainWindow.
class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.turn = 1
        self._translate = QtCore.QCoreApplication.translate
        self.coor = (0,0)
        self.p1_matrix = np.zeros((9,9), dtype=int)
        self.p2_matrix = np.zeros((9,9), dtype=int)
        self.p1_color_matrix = np.zeros((9,9), dtype=tuple)
        self.p2_color_matrix = np.zeros((9,9), dtype=tuple)
        for i in range(9):
                for j in range(9):
                        self.p1_color_matrix[i][j] = WHITE
                        self.p2_color_matrix[i][j] = WHITE
        self.patch = np.array((0,0))

        self.inventory = {1:{'buttons':5, 'space':0, 'income':0}, 2:{'buttons':5, 'space':0, 'income':0}}

        self.pieces = [i for i in range(4,35)]
        shuffle(self.pieces)
        self.pieces.insert(0,3)

        self.MoveAheadButton_p1.clicked.connect(self.move_and_collect)
        self.MoveAheadButton_p2.clicked.connect(self.move_and_collect)
        self.PlacePieceButton_p1.clicked.connect(self.place_piece)
        self.PlacePieceButton_p2.clicked.connect(self.place_piece)
        self.DownloadInstructionsButton_p1.clicked.connect(self.download_instructions)
        self.DownloadInstructionsButton_p2.clicked.connect(self.download_instructions)

        self.load_pieces()
        self.create_shapes()


    def load_pieces(self):
        images_names = os.listdir('images/')
        self.piece_info = pd.read_csv('piece_data.csv').to_dict('index')
        
        for key in reversed(list(self.piece_info.keys())):
            self.piece_info[key+3] = self.piece_info[key] # type: ignore
        del self.piece_info[0]
        del self.piece_info[1]
        del self.piece_info[2]
        for key in self.piece_info.keys():
            self.piece_info[key]['color'] = hex_to_rgb(self.piece_info[key]['color'])
            if key < 10: # type: ignore
                piece_num = "0" + str(key)
            else:
                piece_num = str(key)
            for name in images_names:
                if name.startswith(piece_num):
                    self.piece_info[key]['image'] = 'images/' + name

        # for key in time_space:
        #     time_space[key]['contents'] = time_space[key]['space_coor']

        self.piece_info[3]['shape'] = np.array(([1,1]))
        self.piece_info[4]['shape'] = np.array(([1,1], [0,1]))
        self.piece_info[5]['shape'] = np.array(([1,1], [0,1]))
        self.piece_info[6]['shape'] = np.array(([1,1,1]))
        self.piece_info[7]['shape'] = np.array(([1,1], [1,1]))
        self.piece_info[8]['shape'] = np.array(([1,1,1,1]))
        self.piece_info[9]['shape'] = np.array(([1,1,1], [1,0,0]))
        self.piece_info[10]['shape'] = np.array(([1,1,1], [1,0,0]))
        self.piece_info[11]['shape'] = np.array(([1,1,1], [0,1,0]))
        self.piece_info[12]['shape'] = np.array(([0,1,1], [1,1,0]))
        self.piece_info[13]['shape'] = np.array(([0,1,1], [1,1,0]))
        self.piece_info[14]['shape'] = np.array(([0,1,0], [1,1,1], [0,1,0]))
        self.piece_info[15]['shape'] = np.array(([1,0,1], [1,1,1]))
        self.piece_info[16]['shape'] = np.array(([1,1,1], [1,1,0]))
        self.piece_info[17]['shape'] = np.array(([1,1,1], [0,1,0], [0,1,0]))
        self.piece_info[18]['shape'] = np.array(([0,1,0,0], [1,1,1,1]))
        self.piece_info[19]['shape'] = np.array(([1,1,0,0], [0,1,1,1]))
        self.piece_info[20]['shape'] = np.array(([1,0,0], [1,1,0], [0,1,1]))
        self.piece_info[21]['shape'] = np.array(([1,0,0,0], [1,1,1,1]))
        self.piece_info[22]['shape'] = np.array(([0,1,1], [1,1,0], [0,1,1]))
        self.piece_info[23]['shape'] = np.array(([1,1,0,0], [1,1,1,1]))
        self.piece_info[24]['shape'] = np.array(([0,1,1,0], [1,1,1,1]))
        self.piece_info[25]['shape'] = np.array(([0,1,0,0], [1,1,1,1], [0,1,0,0]))
        self.piece_info[26]['shape'] = np.array(([0,0,1,0], [1,1,1,1], [0,1,0,0]))
        self.piece_info[27]['shape'] = np.array(([1,1,0], [1,1,0], [0,1,1]))
        self.piece_info[28]['shape'] = np.array(([1,0,0,1], [1,1,1,1]))
        self.piece_info[29]['shape'] = np.array(([1,0,0,0], [1,1,1,1], [0,0,0,1]))
        self.piece_info[30]['shape'] = np.array(([0,0,0,1], [1,1,1,1], [0,0,0,1]))
        self.piece_info[31]['shape'] = np.array(([0,1,1,1], [1,1,1,0]))
        self.piece_info[32]['shape'] = np.array(([0,0,1,0,0], [1,1,1,1,1], [0,0,1,0,0]))
        self.piece_info[33]['shape'] = np.array(([1,1,1], [0,1,0], [1,1,1]))
        self.piece_info[34]['shape'] = np.array(([0,1,1,0], [1,1,1,1], [0,1,1,0]))

    def ones_to_colors(self, matrix):
        temp = matrix.copy().tolist()
        try:
            shape = (len(temp), len(temp[0]))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if matrix[i][j] == 1:
                        temp[i][j] = self.patch_color
                    elif matrix[i][j] == 0:
                        temp[i][j] = WHITE
        except TypeError:
            shape = (len(temp), 0)
            for i in range(shape[0]):
                if matrix[i] == 1:
                    temp[i] = self.patch_color
                elif matrix[i] == 0:
                    temp[i] = WHITE
        return np.array(temp, dtype="i,i,i")
    
    def keyPressEvent(self, event):
        # if a certain button is pressed, move in that direction
        if len(self.patch.shape) == 1:
            self.patch = self.patch[np.newaxis]

        elif event.key() == Qt.Key_S:
            if self.coor[0] + self.patch.shape[0] != 9:
                self.coor = (self.coor[0] + 1, self.coor[1]) # move down

        elif event.key() == Qt.Key_D:
            if self.coor[1] + self.patch.shape[1] != 9:
                self.coor = (self.coor[0], self.coor[1] + 1) # move right
        
        elif event.key() == Qt.Key_W:
            if self.coor[0] != 0:
                self.coor = (self.coor[0] - 1, self.coor[1]) # move up

        elif event.key() == Qt.Key_A:
            if self.coor[1] != 0:
                self.coor = (self.coor[0], self.coor[1] - 1) # move left

        elif event.key() == Qt.Key_R:
            if self.coor[1] + self.patch.shape[0] > 9:
                return
            if self.coor[0] + self.patch.shape[1] > 9:
                return
            self.patch = np.rot90(self.patch, k = -1)

        elif event.key() == Qt.Key_E:
            if self.coor[1] + self.patch.shape[0] > 9:
                return
            if self.coor[0] + self.patch.shape[1] > 9:
                return
            self.patch = np.rot90(self.patch, k = 1)

        elif event.key() == Qt.Key_Q:
            self.patch = np.flipud(self.patch)

        self.revert_to_colors()
        self.color_quilt_with_patch(self.patch_color)

    def check(self):
          
        # checking if it is checked
        if self.option1.isChecked():
            self.piece_num = self.available_shapes[0]
            self.option_num = 0
        elif self.option2.isChecked():
            self.piece_num = self.available_shapes[1]
            self.option_num = 1
        elif self.option3.isChecked():
            self.piece_num = self.available_shapes[2]
            self.option_num = 2
        else:
            return

        self.revert_to_colors()
        self.patch = self.piece_info[self.piece_num]['shape']
        self.coor = (0,0)
        self.patch_color = self.piece_info[self.piece_num]['color']
        self.color_quilt_with_patch(self.patch_color)
    
    def can_afford(self):
        cost = self.piece_info[self.piece_num]['cost']
        buttons = self.inventory[self.turn]['buttons']
        if cost <= buttons:
            return True
        else:
            return False
    
    def buy_piece(self):
        cost = self.piece_info[self.piece_num]['cost']
        spaces = self.piece_info[self.piece_num]['time']
        income = self.piece_info[self.piece_num]['income']
        self.inventory[self.turn]['income'] += income
        self.inventory[self.turn]['buttons'] -= cost
        self.inventory[self.turn]['space'] += spaces
        self.pieces.remove(self.piece_num)
        self.pieces = rotate(self.pieces,self.option_num)
        self.delete_shapes()
        self.create_shapes()

        self.inventory_p2.setText(self._translate("MainWindow", "Income: 0\nButtons: 5\nTime Spaces\nRemaining: 54"))

    def receive_income(self):
        self.buttons += self.income
    
    def move_and_collect(self):
        
        pass

    def place_piece(self):
        if 2 in self.add_patch_to_matrix():
            self.ErrorMessage_p1.setText(self._translate("MainWindow", "Invalid Placement"))
            self.ErrorMessage_p2.setText(self._translate("MainWindow", "Invalid Placement"))
            return

        if not self.can_afford():
            self.ErrorMessage_p1.setText(self._translate("MainWindow", "Cannot Afford Piece"))
            self.ErrorMessage_p2.setText(self._translate("MainWindow", "Cannot Afford Piece"))
            return

        if self.turn == 1:
            self.p1_matrix = self.add_patch_to_matrix()
            self.p1_color_matrix = self.add_colors_to_color_matrix(self.p1_color_matrix, self.ones_to_colors(self.patch))
            
        else:
            self.p2_matrix = self.add_patch_to_matrix()
            self.p2_color_matrix = self.add_colors_to_color_matrix(self.p2_color_matrix, self.ones_to_colors(self.patch))
        
        self.coor = (0,0)
        self.ErrorMessage_p1.setText(self._translate("MainWindow", ""))
        self.ErrorMessage_p2.setText(self._translate("MainWindow", ""))

        # print(self.p1_color_matrix, self.p2_color_matrix)
        self.buy_piece()
        self.update_turn()

    def download_instructions(self):
        pass        
    
    def current_quilt(self):
        if self.turn == 1:
            return self.p1quilt
        else:
            return self.p2quilt

    def current_matrix(self):
        if self.turn == 1:
            return self.p1_matrix
        else:
            return self.p2_matrix

    def current_color_matrix(self):
        if self.turn == 1:
            return self.p1_color_matrix
        else:
            return self.p2_color_matrix

    def change_tabs(self, num):
        print("tab should change")
        self.MainWindow_2.setCurrentIndex(num)

    def color_cell(self, coor:tuple, color:tuple):
            item = QtWidgets.QTableWidgetItem()
            brush = QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2]))
            brush.setStyle(QtCore.Qt.SolidPattern) # type: ignore
            item.setBackground(brush)
            self.current_quilt().setItem(coor[0], coor[1], item) 
    
    def add_patch_to_matrix(self):
        temp_matrix = self.current_matrix().copy()

        temp_patch = self.patch

        if len(temp_patch.shape) == 2:
            temp_matrix[self.coor[0]:self.coor[0] + temp_patch.shape[0], self.coor[1]:self.coor[1] + temp_patch.shape[1]] += temp_patch  # use array slicing
        else:
            temp_matrix[self.coor[0]:self.coor[0] + 1, self.coor[1]:self.coor[1] + len(temp_patch)][0] += temp_patch  # use array slicing
        return temp_matrix

    def add_colors_to_color_matrix(self, current_matrix, temp_patch):
        temp_matrix = current_matrix.copy()
        if len(temp_patch.shape) == 2:
            temp_matrix[self.coor[0]:self.coor[0] + temp_patch.shape[0], self.coor[1]:self.coor[1] + temp_patch.shape[1]] = temp_patch  # use array slicing
        else:
            temp_matrix[self.coor[0]:self.coor[0] + 1, self.coor[1]:self.coor[1] + len(temp_patch)][0] = temp_patch  # use array slicing
        return temp_matrix


    def color_quilt_with_patch(self, color:tuple):
            temp_matrix = self.add_patch_to_matrix()
            if len(self.patch.shape) == 2:
                for i in range(self.patch.shape[1]):
                    for j in range(self.patch.shape[0]):
                        if temp_matrix[j+self.coor[0]][i+self.coor[1]] > 0 and self.patch[j][i] == 1:
                            self.color_cell((j+self.coor[0], i+self.coor[1]), color)
            else:
                for i, _ in enumerate(self.patch):
                    self.color_cell((self.coor[0], i+self.coor[1]), color)

            for i in range(9):
                    for j in range(9):
                        if temp_matrix[i][j] > 1:
                            self.color_cell((i, j), (0,0,0))

    def revert_to_colors(self):
        for i in range(9):
            for j in range(9):
                self.color_cell((i,j),self.current_color_matrix()[i][j])

    def active_piece(self):
        if self.turn == 1:
            return self.p1_time
        else:
            return self.p2_time
    
    def move_active_piece(self, spaces):
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter) # type: ignore
        icon = QtGui.QIcon()
        item.setIcon(icon)
        if self.turn == 1:
            icon.addPixmap(QtGui.QPixmap("images/p1-piece.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off) # type: ignore
            self.p1_time += spaces
            self.time_board.setItem(time_space['space_coor'][self.p1_time], item)
        else:
            icon.addPixmap(QtGui.QPixmap("images/p2-piece.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off) # type: ignore
            self.p2_time += spaces
            self.time_board.setItem(time_space['space_coor'][self.p2_time], item)

    def create_shapes(self):
        
        self.available_shapes = [self.pieces[0], self.pieces[1], self.pieces[2]]
        self.unavailable_shapes = self.pieces[3:]
        self.unavailable_images = {}
        self.unavailable_info = {}
        self.option1 = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.option2 = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.option3 = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        options = [self.option1, self.option2, self.option3]
        
        for i, shape_num in enumerate(self.available_shapes):
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(self.piece_info[shape_num]['image']), QtGui.QIcon.Normal, QtGui.QIcon.Off)  # type: ignore
            options[i].setIcon(icon1)
            options[i].setIconSize(QtCore.QSize(64, 64))
            options[i].setObjectName("option" + str(i))
            self.verticalLayout_2.addWidget(options[i])
            text = "Cost: " + str(self.piece_info[shape_num]['cost']) + "\nMoves: " + str(self.piece_info[shape_num]['time']) + \
                "\nIncome: " + str(self.piece_info[shape_num]['income'])
            options[i].setText(self._translate("MainWindow", text))

        for i, shape_num in enumerate(self.unavailable_shapes):
            self.unavailable_images[i] = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            self.unavailable_images[i].setEnabled(True)
            self.unavailable_images[i].setMaximumSize(QtCore.QSize(self.piece_info[shape_num]['width'], self.piece_info[shape_num]['height']))
            self.unavailable_images[i].setLayoutDirection(QtCore.Qt.LeftToRight) # type: ignore
            self.unavailable_images[i].setFrameShape(QtWidgets.QFrame.NoFrame)
            self.unavailable_images[i].setText("")
            self.unavailable_images[i].setPixmap(QtGui.QPixmap(self.piece_info[shape_num]['image']))
            self.unavailable_images[i].setScaledContents(True)
            self.unavailable_images[i].setAlignment(QtCore.Qt.AlignCenter) # type: ignore
            self.unavailable_images[i].setObjectName("unavailable_image_" + str(i))
            self.unavailable_info[i] = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            self.unavailable_info[i].setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter) # type: ignore
            self.unavailable_info[i].setObjectName("unavailable_info_" + str(i))
            text = "Cost: " + str(self.piece_info[shape_num]['cost']) + "\nMoves: " + str(self.piece_info[shape_num]['time']) + \
                "\nIncome: " + str(self.piece_info[shape_num]['income'])
            self.unavailable_info[i].setText(self._translate("MainWindow", text))
            self.verticalLayout_2.addWidget(self.unavailable_info[i])
            self.verticalLayout_2.addWidget(self.unavailable_images[i])

            # add check for if radio buttons have been pressed
            self.option1.clicked.connect(self.check)
            self.option2.clicked.connect(self.check)
            self.option3.clicked.connect(self.check)

    def delete_shapes(self):
        
        for i in reversed(range(self.verticalLayout_2.count())):
            self.verticalLayout_2.itemAt(i).widget().setParent(None)

    def update_turn(self):
        self.option1.setChecked(False)
        self.option2.setChecked(False)
        self.option3.setChecked(False)

        self.turn = 3 - self.turn
        if self.turn == 1:
                self.turn_marker_p1.setText(self._translate("MainWindow", "Player 1\'s Turn"))
                self.turn_marker_p2.setText(self._translate("MainWindow", "Player 1\'s Turn"))
                self.turn_marker_board.setText(self._translate("MainWindow", "Player 1\'s Turn"))
                self.change_tabs(0)
        else:
                self.turn_marker_p1.setText(self._translate("MainWindow", "Player 2\'s Turn"))
                self.turn_marker_p2.setText(self._translate("MainWindow", "Player 2\'s Turn"))
                self.turn_marker_board.setText(self._translate("MainWindow", "Player 2\'s Turn"))
                self.change_tabs(1)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MyMainWindow()
    win.show()
    app.exec_()