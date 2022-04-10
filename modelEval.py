import pandas as pd
import numpy as np
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.stats import pearsonr
from tensorflow import keras
import yaml
import DRLstm


class ModelTest:

    def __init__(self, run_id, x_step, y_step):
        """
        测试模型
        :param run_id:
        :param x_step:
        :param y_step:
        """
        # lxq如何归一化？
        # 源域模型 -> 源域数据
        # 目标域模型 -> 目标域训练数据
        # 迁移模型 -> 目标域训练数据
        # 模型测试 -> 目标域训练数据
        # xm1 -> xm1
        # ypj1 -> ypj1 his
        # trans -> xm1
        # test -> ypj1 his
        self.run_path = '.\\test\\run_%s' % run_id

        self.source_data = pd.read_csv('%s\\source_data.csv' % self.run_path, index_col=0, parse_dates=True).iloc[:, 0]
        self.target_data = pd.read_csv('%s\\target_data.csv' % self.run_path, index_col=0, parse_dates=True).iloc[:, 0]
        self.test_data = pd.read_csv('%s\\test_data.csv' % self.run_path, index_col=0, parse_dates=True).iloc[:, 0]
        self.af_data = pd.read_csv('.\\airforecast.csv', index_col=0, parse_dates=True)  # 辅助dr-lstm建模的气象预报数据

        with open('%s\\info.yaml' % self.run_path) as file:
            self.info = yaml.safe_load(file)
        self.param = self.info['param']
        self.coef_dic_source = self.info['coef_dic_source']
        self.coef_dic_target = self.info['coef_dic_target']

        self.source_avg, self.source_std = DRLstm.get_normalization_info(self.source_data)
        self.target_avg, self.target_std = DRLstm.get_normalization_info(self.target_data)

        self.source_model = keras.models.load_model('%s\\source_model.h5' % self.run_path)
        self.target_model = keras.models.load_model('%s\\target_model.h5' % self.run_path)
        self.trans_model = keras.models.load_model('%s\\trans_model.h5' % self.run_path)

        self.source_dr_model = DRLstm.DRLstm(self.test_data, self.af_data,
                                             self.source_model,
                                             self.source_avg, self.source_std,  # lxq源域dr_lstm模型,用源域数据标准化;
                                             self.param,
                                             self.coef_dic_source)  # lxq源域系数

        self.target_dr_model = DRLstm.DRLstm(self.test_data, self.af_data,
                                             self.target_model,
                                             self.target_avg, self.target_std,  # 目标域dr_lstm模型
                                             self.param,
                                             self.coef_dic_target)  # lxq目标域系数

        self.trans_dr_model = DRLstm.DRLstm(self.test_data, self.af_data,
                                            self.trans_model,
                                            self.source_avg, self.source_std,
                                            self.param,
                                            self.coef_dic_target)  # 目标域迁移dr_lstm模型, lxq目标域系数

        self.x_step = x_step
        self.y_step = y_step

        # self.test_x = None
        # self.test_y = None
        self.source_x, self.source_y = None, None
        self.target_x, self.target_y = None, None

        self.source_pred = None  # 预测结果
        self.target_pred = None
        self.trans_pred = None
        self.source_dr_pred = None
        self.target_dr_pred = None
        self.trans_dr_pred = None
        self.pure_pred = None
        self.mix_pred = None

    def create_test_dataset(self, interpolate=True, normalization=True):
        if interpolate:
            self.test_data.interpolate(inplace=True)

        if normalization:  # yyq:测试时用什么数据进行归一化？
            source_input = (self.test_data - self.source_avg) / self.source_std  # 用源域数据对测试数据归一化
            target_input = (self.test_data - self.target_avg) / self.target_std  # 用目标域数据对测试数据归一化
            # data = (self.test_data - self.target_avg) / self.target_std
            # data = (self.test_data - self.source_avg) / self.source_std
            # mms = MinMaxScaler()
            # data = mms.fit_transform(np.array(self.test_data).reshape(-1, 1))
        else:
            source_input = self.test_data
            target_input = self.test_data

        source_x, source_y = [], []
        target_x, target_y = [], []

        for index in range(len(source_input) - (self.x_step + self.y_step - 1)):
            source_x.append(source_input[index:index + self.x_step])
        self.source_x = np.array(source_x).squeeze()

        for index in range(len(source_input) - (self.x_step + self.y_step - 1)):
            source_y.append(source_input[index + self.x_step:index + self.x_step + self.y_step])
        self.source_y = np.array(source_y).squeeze() * self.source_std + self.source_avg  # yyq:测试时用什么数据进行归一化？
        # self.test_y = np.array(data_y).squeeze() * self.source_std + self.source_avg

        for index in range(len(target_input) - (self.x_step + self.y_step - 1)):
            target_x.append(target_input[index:index + self.x_step])
        self.target_x = np.array(target_x).squeeze()

        for index in range(len(target_input) - (self.x_step + self.y_step - 1)):
            target_y.append(target_input[index + self.x_step:index + self.x_step + self.y_step])
        self.target_y = np.array(target_y).squeeze() * self.target_std + self.target_avg  # yyq:测试时用什么数据进行归一化？

    def model_pred(self):
        self.source_pred = self.source_model.predict(self.source_x) * self.source_std + self.source_avg  # lxq反归一化
        self.target_pred = self.target_model.predict(self.target_x) * self.target_std + self.target_avg
        self.trans_pred = self.trans_model.predict(self.source_x) * self.source_std + self.source_avg
        self.source_dr_pred = self.source_dr_model.predict()
        self.target_dr_pred = self.target_dr_model.predict()
        self.trans_dr_pred = self.trans_dr_model.predict()
        self.pure_pred = self.source_dr_model.pure_pred
        self.mix_pred = self.source_dr_model.mix_pred

    def eval_rmse(self, days):
        rmse = {'source': ModelTest.cal_rmse(self.source_y[:, days - 1], self.trans_pred[:, days - 1]),
                'target': ModelTest.cal_rmse(self.target_y[:, days - 1], self.target_pred[:, days - 1]),
                'trans': ModelTest.cal_rmse(self.source_y[:, days - 1], self.source_pred[:, days - 1]),
                'source-dr': ModelTest.cal_rmse(self.source_y[:, days - 1], self.trans_dr_pred[:, days - 1]),
                'target-dr': ModelTest.cal_rmse(self.target_y[:, days - 1], self.target_dr_pred[:, days - 1]),
                'trans-dr': ModelTest.cal_rmse(self.source_y[:, days - 1], self.source_dr_pred[:, days - 1]),
                'pure': ModelTest.cal_rmse(self.source_y[:, days - 1], self.pure_pred[:, days - 1]),
                'mix': ModelTest.cal_rmse(self.source_y[:, days - 1], self.mix_pred[:, days - 1])}
        return rmse

    def eval_r2(self, days):
        r2 = {'source': ModelTest.cal_r2(self.source_y[:, days - 1], self.trans_pred[:, days - 1]),
              'target': ModelTest.cal_r2(self.target_y[:, days - 1], self.target_pred[:, days - 1]),
              'trans': ModelTest.cal_r2(self.source_y[:, days - 1], self.source_pred[:, days - 1]),
              'source-dr': ModelTest.cal_r2(self.source_y[:, days - 1], self.trans_dr_pred[:, days - 1]),
              'target-dr': ModelTest.cal_r2(self.target_y[:, days - 1], self.target_dr_pred[:, days - 1]),
              'trans-dr': ModelTest.cal_r2(self.source_y[:, days - 1], self.source_dr_pred[:, days - 1]),
              'pure': ModelTest.cal_r2(self.source_y[:, days - 1], self.pure_pred[:, days - 1]),
              'mix': ModelTest.cal_r2(self.source_y[:, days - 1], self.mix_pred[:, days - 1])}
        return r2

    def eval_mae(self, days):
        mae = {'source': ModelTest.cal_mae(self.source_y[:, days - 1], self.trans_pred[:, days - 1]),
               'target': ModelTest.cal_mae(self.target_y[:, days - 1], self.target_pred[:, days - 1]),
               'trans': ModelTest.cal_mae(self.source_y[:, days - 1], self.source_pred[:, days - 1]),
               'source-dr': ModelTest.cal_mae(self.source_y[:, days - 1], self.trans_dr_pred[:, days - 1]),
               'target-dr': ModelTest.cal_mae(self.target_y[:, days - 1], self.target_dr_pred[:, days - 1]),
               'trans-dr': ModelTest.cal_mae(self.source_y[:, days - 1], self.source_dr_pred[:, days - 1]),
               'pure': ModelTest.cal_mae(self.source_y[:, days - 1], self.pure_pred[:, days - 1]),
               'mix': ModelTest.cal_mae(self.source_y[:, days - 1], self.mix_pred[:, days - 1])}
        return mae

    def eval_mape(self, days):
        mape = {'source': ModelTest.cal_mape(self.source_y[:, days - 1], self.trans_pred[:, days - 1]),
                'target': ModelTest.cal_mape(self.target_y[:, days - 1], self.target_pred[:, days - 1]),
                'trans': ModelTest.cal_mape(self.source_y[:, days - 1], self.source_pred[:, days - 1]),
                'source-dr': ModelTest.cal_mape(self.source_y[:, days - 1], self.trans_dr_pred[:, days - 1]),
                'target-dr': ModelTest.cal_mape(self.target_y[:, days - 1], self.target_dr_pred[:, days - 1]),
                'trans-dr': ModelTest.cal_mape(self.source_y[:, days - 1], self.source_dr_pred[:, days - 1]),
                'pure': ModelTest.cal_mape(self.source_y[:, days - 1], self.pure_pred[:, days - 1]),
                'mix': ModelTest.cal_mape(self.source_y[:, days - 1], self.mix_pred[:, days - 1])}
        return mape

    def eval_acc(self, days):
        acc = {'source': ModelTest.cal_acc(self.source_y[:, days - 1], self.trans_pred[:, days - 1], self.param),
               'target': ModelTest.cal_acc(self.target_y[:, days - 1], self.target_pred[:, days - 1], self.param),
               'trans': ModelTest.cal_acc(self.source_y[:, days - 1], self.source_pred[:, days - 1], self.param),
               'source-dr': ModelTest.cal_acc(self.source_y[:, days - 1], self.trans_dr_pred[:, days - 1], self.param),
               'target-dr': ModelTest.cal_acc(self.target_y[:, days - 1], self.target_dr_pred[:, days - 1], self.param),
               'trans-dr': ModelTest.cal_acc(self.source_y[:, days - 1], self.source_dr_pred[:, days - 1], self.param),
               'pure': ModelTest.cal_acc(self.source_y[:, days - 1], self.pure_pred[:, days - 1], self.param),
               'mix': ModelTest.cal_acc(self.source_y[:, days - 1], self.mix_pred[:, days - 1], self.param)}
        return acc

    def eval_model(self):
        result_rmse = {}
        result_r2 = {}
        result_mae = {}
        result_mape = {}
        result_acc = {}
        for day in range(1, self.y_step + 1):
            result_rmse[day] = self.eval_rmse(day)
            result_r2[day] = self.eval_r2(day)
            result_mae[day] = self.eval_mae(day)
            result_mape[day] = self.eval_mape(day)
            result_acc[day] = self.eval_acc(day)

        return pd.DataFrame(result_rmse), pd.DataFrame(result_r2), pd.DataFrame(result_mae), pd.DataFrame(
            result_mape), pd.DataFrame(result_acc)

    @staticmethod
    def get_level(value, param):
        # lxq等级怎么进行等级评价？
        # 先不考虑
        if param == 'tem_max':
            if value > 32:
                return 3
            elif value > 30:
                return 2
            elif value > 28:
                return 1
            else:
                return 0
        elif param == 'tem_min':
            if value < 10:
                return 3
            elif value < 12:
                return 2
            elif value < 14:
                return 1
            else:
                return 0

    @staticmethod
    def cal_rmse(obs, pred):
        n = len(obs)
        mse = sum(np.square(obs - pred)) / n
        rmse = math.sqrt(mse)
        return rmse

    @staticmethod
    def cal_r2(obs, pred):
        return pearsonr(pred, obs)[0]

    @staticmethod
    def cal_mae(obs, pred):
        n = len(obs)
        mae = sum(np.abs(obs - pred)) / n
        return mae

    @staticmethod
    def cal_mape(obs, pred):
        n = len(obs)
        mape = sum(np.abs((obs - pred) / obs)) / n * 100
        return mape

    @staticmethod
    def cal_acc(obs, pred, param):
        obs_level = list(map(lambda x: ModelTest.get_level(x, param), obs))
        pred_level = list(map(lambda x: ModelTest.get_level(x, param), pred))
        t = 0
        f = 0
        for i in range(len(pred_level)):
            if pred_level[i] == obs_level[i]:
                t += 1
            else:
                f += 1
        return t / (t + f)


if __name__ == '__main__':
    # mt = ModelTest('渔排基回归系数', 27, 7)
    # mt.create_test_dataset()
    # mt.model_pred()
    # print(mt.source_dr_model.weight)
    obs = pd.Series([50, 34, 29, 16, 28, -100])
    pred = pd.Series([40, 39, 23, 12, 27, 20])
    acc = ModelTest.cal_acc(obs, pred)
    print(acc)
