import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class BuildModel:

    def __init__(self, data, name, avg, std, x_step, y_step, split=0.8):
        """
        训练新模型
        :param data: 训练数据,Series格式
        :param name:站点名称 ,(str)
        :param x_step: 输入步长，默认27
        :param y_step: 输出步长，默认7
        :param split: 数据集划分比例
        """
        # assert not os.path.exists('.\\model\\source\\%s.h5' % name)
        self.data = data
        self.name = name

        self.avg = avg
        self.std = std

        self.x_step = x_step
        self.y_step = y_step
        self.split = split

        self.model = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.history = None

    def create_dataset(self, normalization=True, interpolate=True):
        if interpolate:
            self.data.interpolate(inplace=True)

        if normalization:
            # mms = MinMaxScaler()
            # data = mms.fit_transform(np.array(self.data).reshape(-1, 1))

            data = (self.data - self.avg) / self.std
        else:
            data = self.data

        data_x = []
        data_y = []

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_x.append(data[index:index + self.x_step])
        data_x = np.array(data_x)

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_y.append(data[index + self.x_step:index + self.x_step + self.y_step])
        data_y = np.array(data_y)

        self.train_x = data_x[:int(len(data) * self.split)]
        self.test_x = data_x[int(len(data) * self.split):]
        self.train_y = data_y[:int(len(data) * self.split)]
        self.test_y = data_y[int(len(data) * self.split):]

    def build_model(self):
        # lxq模型结构需调整
        # 把模型结构信息加入到run文件夹中
        inputs = Input(shape=(self.x_step, 1))
        lstm1 = LSTM(48, return_sequences=True, name='lstm1')(inputs)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(32, name='lstm2')(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        output = Dense(self.y_step)(dropout2)

        # lstm1 = LSTM(32, activation='relu', kernel_initializer='random_uniform', return_sequences=True, name='lstm1')(
        #     inputs)
        # lstm2 = LSTM(16, activation='relu', kernel_initializer='random_uniform', name='lstm2')(lstm1)
        # x = Dropout(0.2)(lstm2)
        # output = Dense(self.y_step, activation='relu', kernel_initializer='random_uniform')(x)

        model = Model(inputs=inputs, outputs=output)
        self.model = model

        # self.model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])
        self.model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])

    def train_model(self):
        # cb_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        #            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.8, min_lr=0.000001)]
        #
        # self.history = self.model.fit(
        #                             self.train_x,
        #                             self.train_y,
        #                             batch_size=128,
        #                             epochs=100,
        #                             validation_split=0.1,
        #                             callbacks=cb_list,
        #                             verbose=1
        #                             )

        self.history = self.model.fit(
            self.train_x,
            self.train_y,
            batch_size=32,
            epochs=150,
            validation_split=0.1,
            verbose=1
        )

    def save_model(self):
        self.model.save('.\\model\\base\\%s.h5' % self.name)

    def run(self):
        self.create_dataset()
        self.build_model()
        self.train_model()
        self.save_model()
