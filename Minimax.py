import copy

import game

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from keras.layers import Flatten, BatchNormalization
import numpy as np

class NN:
    def __init__(self) -> None:
        self.model = Sequential()

        self.model.add(Conv2D(filters=64, kernel_size=1, activation='relu', input_shape= (5,5,1)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=24, kernel_size=1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=10, kernel_size=1, activation='relu'))
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(2, activation = "softmax"))
        
        self.model.load_weights('f.h5')
    def getRes(self, board):
        arr = np.array(board)
        arr = np.expand_dims(arr, 0)
        arr = np.expand_dims(arr, -1)
        return self.model.predict(arr)
        
class Solver:
    def __init__(self,
                 depth: int = 2,
                 board: list = None,
                 player: int = 1,):
        self.depth = depth
        self.board = copy.deepcopy(board)
        self.player, self.opponent = player, -1 * player
        
        self.start = None
        self.end = None
        self.NN = NN()
    
    def evaluate(self, board):
        # result = sum(map(sum, board))
        # if self.player == -1:
        #     result *= -1
        # return result
        result = self.NN.getRes(board)
        res = result[0][0] - result[0][1]
        if self.player == -1:
            res *= -1
        return res
    
    def play(self, node, dp):
        if dp > self.depth: 
            return
        
        # LEAF NODE
        if dp == self.depth:
            return self.evaluate(node.board)
        
        score = 0
        g = False
        cg = game.CoGanh()
        # PLAYER
        if dp % 2 == 0:
            score = -100
            successor = []
            pos = cg.getPosition(node.board, self.player)
                
            for p in pos:
                successor += cg.move_gen(node, p)
                
            if len(successor) > 0:
                for s in successor:
                    if s[2]:
                        g = True
                        break
                    
                for s in successor:
                    if g:
                        if not s[2]:
                            continue
                        
                    if cg.X_win(s[0].board):
                        if dp == 0:
                            self.start = s[3]
                            self.end = s[1]
                        return 100
                    
                    value = self.play(s[0], dp + 1)
                    if value > score:
                        score = value
                        if dp == 0:
                            self.start = s[3]
                            self.end = s[1]
        # OPPONENT
        else:
            score = 100
            successor = []
            pos = cg.getPosition(node.board, self.opponent)
                
            for p in pos:
                successor += cg.move_gen(node, p)
                
            if len(successor) > 0:
                for s in successor:
                    if s[2]:
                        g = True
                        break
                    
                for s in successor:
                    if g:
                        if not s[2]:
                            continue
                        
                    if cg.O_win(s[0].board): 
                        return -100
                    
                    value = self.play(s[0], dp + 1)
                    if value < score:
                        score = value
                        
        return score
    
    def solv(self):
        node = game.Node_1(self.board)
        score = self.play(node, 0)
        return (self.start, self.end)