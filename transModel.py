import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import copy
import json


class TransModel:

    def __init__(self, model, data, name, avg, std, x_step, y_step):
        """
        迁移模型
        :param model: 源域模型
        :param data: 目标域数据(Series)
        :param name: 站点名称（str）
        :param x_step:
        :param y_step:
        """
        # assert not os.path.exists('.\\model\\trans\\%s.h5' % name)
        # self.model = copy.deepcopy(model)
        self.model = model

        self.data = data
        self.name = name

        self.avg = avg
        self.std = std

        self.x_step = x_step
        self.y_step = y_step

        self.train_x = None
        self.train_y = None

        self.history = None

    def create_dataset(self, normalization=True, interpolate=True):
        if interpolate:
            self.data.interpolate(inplace=True)

        if normalization:
            data = (self.data - self.avg) / self.std
            # mms = MinMaxScaler()
            # data = mms.fit_transform(np.array(self.data).reshape(-1, 1))
        else:
            data = self.data

        data_x = []
        data_y = []

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_x.append(data[index:index + self.x_step])
        self.train_x = np.array(data_x)

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_y.append(data[index + self.x_step:index + self.x_step + self.y_step])
        self.train_y = np.array(data_y)

    def trans_model(self):
        # lxq是否要多加几层模型，从而选取效果最好的冻结方案
        self.model.layers[1].trainable = False  # 冻结第一层

        # cb_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
        #            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.8, min_lr=0.000001)]
        #
        # self.history = self.model.fit(
        #     self.train_x,
        #     self.train_y,
        #     batch_size=128,
        #     epochs=100,
        #     validation_split=0.1,
        #     callbacks=cb_list,
        #     verbose=1
        # )

        # filepath = '.\\model\\trans\\%s.h5' % self.name
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

        self.history = self.model.fit(
            self.train_x,
            self.train_y,
            batch_size=32,
            epochs=150,
            validation_split=0.1,
            # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
            # callbacks=[checkpoint],
            verbose=1
        )

    def save_model(self):
        self.model.save('.\\model\\trans\\%s.h5' % self.name)

    def run(self):
        self.create_dataset()
        self.trans_model()
        self.save_model()
