import datetime
import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import modelEval as mE


class DRLstm:

    def __init__(self, test_data, af_data, lstm_model, avg, std, param,
                 coef_dic,  # dic,包含pure_coef_max, mix_coef_max, pure_coef_min, mix_coef_min  4个值为列表的键
                 x_step=27, y_step=7):
        self.test_data = test_data
        self.af_data = af_data
        self.lstm_model = lstm_model
        self.x_step = x_step
        self.y_step = y_step

        self.avg = avg
        self.std = std
        self.param = param

        self.pure_coef_max = coef_dic['pure_coef_max']  # list, [0.2, 0.1, 0.1, 0.2]
        self.mix_coef_max = coef_dic['mix_coef_max']  # list, [0.5, 0.2, 0.2]
        self.pure_coef_min = coef_dic['pure_coef_min']  # list, [0.2, 0.1, 0.1, 0.2]
        self.mix_coef_min = coef_dic['mix_coef_min']  # list, [0.5, 0.2, 0.2]

        self.test_csv = {}  # 所有类test_csv对象的集合
        self.lstm_input = None
        self.test_y = None

        self.mix_pred = None  # 混差分预测结果
        self.pure_pred = None  # 纯差分预测结果
        self.lstm_pred = None  # lstm预测结果
        self.weighted_pred = None  # 加权预测结果

        self.weight = {'pure_weight': 1,
                       'mix_weight': 1,
                       'lstm_weight': 1}  # lxq如何进行权重调整？  # 参考实际运行流程

    def create_lstm_input(self, interpolate=True, normalization=True):
        if interpolate:
            self.test_data.interpolate(inplace=True)

        if normalization:
            data = (self.test_data - self.avg) / self.std
            # mms = MinMaxScaler()
            # data = mms.fit_transform(np.array(self.test_data).reshape(-1, 1))
        else:
            data = self.test_data

        data_x = []
        data_y = []

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_x.append(data[index:index + self.x_step])
        self.lstm_input = np.array(data_x).squeeze()

        for index in range(len(data) - (self.x_step + self.y_step - 1)):
            data_y.append(data[index + self.x_step:index + self.x_step + self.y_step])
        self.test_y = np.array(data_y).squeeze() * self.std + self.avg

    def create_dr_input(self):
        for index in range(len(self.test_data) - (self.x_step + self.y_step - 1)):
            # self.input_data[index] = {}
            # self.input_data[index]['data_lstm'] = np.array(data[index:index + self.x_step]).squeeze()
            end2 = self.test_data.index[index + self.x_step].date()  # 预测日日期
            end1 = end2 + datetime.timedelta(days=-1)  # 昨天
            start = end2 + datetime.timedelta(days=-3)  # 3天前
            wt = self.test_data.loc[start:end1]  # 3天的水温
            at = self.af_data.loc[start:end2, :]
            self.test_csv[index] = pd.concat([wt, at], axis=1)

    def mix_predict(self):
        mix_pred = {}
        if not self.test_csv:
            self.create_dr_input()
        for index, test_csv in self.test_csv.items():
            pred = getmixpredict(test_csv, self.param,
                                 self.mix_coef_max,
                                 self.mix_coef_min)
            mix_pred[index] = pred
        # return np.array(pd.DataFrame(self.mix_pred).T.sort_index())
        self.mix_pred = np.array(pd.DataFrame(mix_pred).T.sort_index())

    def pure_predict(self):
        pure_pred = {}
        if not self.test_csv:
            self.create_dr_input()
        for index, test_csv in self.test_csv.items():
            pred = getpurepredict(test_csv, self.param,
                                  self.pure_coef_max,
                                  self.pure_coef_min)
            pure_pred[index] = pred
        # return np.array(pd.DataFrame(self.pure_pred).T.sort_index())
        self.pure_pred = np.array(pd.DataFrame(pure_pred).T.sort_index())

    def lstm_predict(self):
        if self.lstm_input is None:
            self.create_lstm_input()
        self.lstm_pred = self.lstm_model.predict(self.lstm_input) * self.std + self.avg
        # return self.lstm_pred

    def weighted_predict(self):
        weighted_pred = {}
        for index in range(len(self.test_y)):
            obs = self.test_y[index, :]
            pure_pred = self.pure_pred[index, :]
            mix_pred = self.mix_pred[index, :]
            lstm_pred = self.lstm_pred[index, :]
            # pure_rmse = mE.ModelTest.cal_rmse(obs, pure_pred)
            # mix_rmse = mE.ModelTest.cal_rmse(obs, mix_pred)
            # lstm_rmse = mE.ModelTest.cal_rmse(obs, lstm_pred)
            pure_diff = abs(obs[0] - pure_pred[0])  # lxq根据第一天的差值来比较
            mix_diff = abs(obs[0] - mix_pred[0])
            lstm_diff = abs(obs[0] - lstm_pred[0])

            pure_weighted_pred = pure_pred * self.weight['pure_weight']
            mix_weighted_pred = mix_pred * self.weight['mix_weight']
            lstm_weighted_pred = lstm_pred * self.weight['lstm_weight']
            sum_pred = pure_weighted_pred + mix_weighted_pred + lstm_weighted_pred
            sum_weight = self.weight['pure_weight'] + self.weight['mix_weight'] + self.weight['lstm_weight']
            weighted_pred[index] = sum_pred / sum_weight
            if min([pure_diff, mix_diff, lstm_diff]) == pure_diff:
                self.weight['pure_weight'] += 1
            elif min([pure_diff, mix_diff, lstm_diff]) == mix_diff:
                self.weight['mix_weight'] += 1
            else:
                self.weight['lstm_weight'] += 1
        self.weighted_pred = np.array(pd.DataFrame(weighted_pred).T.sort_index())

    def predict(self):
        self.mix_predict()
        self.pure_predict()
        self.lstm_predict()
        self.weighted_predict()
        return self.weighted_pred


def getdata(test_csv, parameter):
    """
    :param test_csv:格式类似test_csv的DataFrame对象
    :param parameter(str) 'fore_max_1' / 'fore_min_1' / ...
    读取对应站点，列名为parameter的某一列的数据
    返回test.csv的某一列数据--
    getdata('第1天最高气温')
    """
    df = test_csv[parameter]
    return list(df)


def mixd1(test_csv, parameter, coef_max, coef_min):
    # 0.5 0.2 0.2
    """coef_max, coef_min:长度为3的列表"""
    if parameter == 'tem_max':
        nameh = 'fore_max_1'
        datadh = []
        dataneedh = getdata(test_csv, nameh)
        dataneed = getdata(test_csv, parameter)
        datad = float(dataneed[2]) - float(dataneed[1])
        for i in range(3):
            datadh.append(float(dataneedh[i + 1]) - float(dataneedh[i]))

        if datad == 0 and abs(datadh[2]) <= 1 and datadh[1] == 0 or datadh[2] == 0 and datadh[1] == 0:
            data = 0
        else:
            data = coef_max[0] * datad + coef_max[1] * datadh[2]

    if parameter == 'tem_min':
        namel = 'fore_min_1'
        datadl = []
        dataneedl = getdata(test_csv, namel)
        dataneed = getdata(test_csv, parameter)
        datad = float(dataneed[2]) - float(dataneed[1])
        for i in range(3):
            datadl.append(float(dataneedl[i + 1]) - float(dataneedl[i]))

        if datad == 0 and abs(datadl[2]) <= 1 and datadl[1] == 0 or datadl[2] == 0 and datadl[1] == 0:
            data = 0
        else:
            data = coef_min[0] * datad + coef_min[1] * datadl[2]
    return data


def mixd2(test_csv, parameter, coef_max, coef_min):
    """coef_max, coef_min:长度为3的列表"""
    datad = []
    data0 = mixd1(test_csv, parameter, coef_max, coef_min)
    if parameter == 'tem_max':
        nameh = 'fore_max_1'
        datadh = []
        for i in range(6):
            namenh = 'fore_max_%s' % (i + 2)
            dataneedh = getdata(test_csv, namenh)
            datah = getdata(test_csv, nameh)
            datadh.append((float(dataneedh[3]) - float(datah[3])) * coef_max[2][i] + data0)
        datad = datadh
    if parameter == 'tem_min':
        namel = 'fore_min_1'
        datadl = []
        for i in range(6):
            namenl = 'fore_min_%s' % (i + 2)
            dataneedl = getdata(test_csv, namenl)
            datal = getdata(test_csv, namel)
            datadl.append((float(dataneedl[3]) - float(datal[3])) * coef_min[2][i] + data0)
        datad = datadl
    return datad


def getmixpredict(test_csv, parameter, coef_max, coef_min):
    """coef_max, coef_min:长度为3的列表"""
    dataneed = getdata(test_csv, parameter)
    datad = []
    datad0 = mixd1(test_csv, parameter, coef_max, coef_min)
    datad2 = mixd2(test_csv, parameter, coef_max, coef_min)
    datad.append(datad0)
    for i in range(6):
        datad.append(datad2[i])
    datapre = []
    for i in range(7):
        datapre.append(float(dataneed[2]) + datad[i])
    return datapre


def pured1(test_csv, parameter, coef_max, coef_min):
    """
    ###根据参数得到第一天的差分组合结果
    返回一个差分值,用于修正第一天的预测结果
    pured1('最高水温')
    coef_max, coef_min:长度为4的列表
    """
    # 0.1 0.1 0.2 0.2
    if parameter == 'tem_max':
        nameh = 'fore_max_1'
        datadh = []  # 第一天最高气温差分列表
        dataneedh = getdata(test_csv, nameh)  # 得到第一天最高气温的数据
        for i in range(3):
            datadh.append(float(dataneedh[i + 1]) - float(dataneedh[i]))  # 得到3个最新的第一天最高气温差分结果
        datah = datadh[0] * coef_max[0] + datadh[1] * coef_max[1] + datadh[2] * coef_max[2]
        data = datah  # 得到最高水温的差分结果
    if parameter == 'tem_min':
        namel = 'fore_min_1'
        datadl = []  # 第一天最低气温差分列表
        dataneedl = getdata(test_csv, namel)  # 得到第一天最低气温的数据
        for i in range(3):
            datadl.append(float(dataneedl[i + 1]) - float(dataneedl[i]))  # 得到3个最新的第一天最低气温差分结果
        datal = datadl[0] * coef_min[0] + datadl[1] * coef_min[1] + datadl[2] * coef_min[2]
        data = datal  # 得到最低水温的差分结果
    return data


def pured2(test_csv, parameter, coef_max, coef_min):
    """
    ##得到第2天至第7天的纯差分
    coef_max, coef_min:长度为4的列表
    """
    # lxq同pured1
    datad = []
    if parameter == 'tem_max':
        nameh = 'fore_max_1'
        datadh = []
        data0 = pured1(test_csv, parameter, coef_max, coef_min)
        datah0 = data0
        for i in range(6):
            namenh = 'fore_max_%s' % (i + 2)
            dataneedh = getdata(test_csv, namenh)  # 得到第i+2天的最高气温
            datah = getdata(test_csv, nameh)  # 得到第1天的最高气温
            datadh.append((float(dataneedh[3]) - float(datah[3])) * coef_max[3][i] + datah0)  # 得到第i+2天的最高水温的差分
        datad = datadh  # 得到最高水温第2天至第7天的差分

    if parameter == 'tem_min':
        namel = 'fore_min_1'
        datadl = []
        data0 = pured1(test_csv, parameter, coef_max, coef_min)
        datal0 = data0
        for i in range(6):
            namenl = 'fore_min_%s' % (i+2)
            dataneedl = getdata(test_csv, namenl)  # 得到第i+2的最低气温
            datal = getdata(test_csv, namel)  # 得到第1天的最低气温
            datadl.append((float(dataneedl[3]) - float(datal[3])) * coef_max[3][i] + datal0)  # 得到第i+2天的最低水温的差分
        datad = datadl  # 得到最低水温第2天至第7天的差分
    return datad


def getpurepredict(test_csv, parameter, coef_max, coef_min):
    """
    返回某指标纯差分预测的结果:[29.5, 28.1, 28.5, 28.9, 28.7, 28.7, 28.9]
    getpurepredict(667048424, 'tem_max')
    返回预测结果列表
    coef_max, coef_min:长度为4的列表
    """
    dataneed = getdata(test_csv, parameter)
    datad = []
    datad0 = pured1(test_csv, parameter, coef_max, coef_min)
    datad2 = pured2(test_csv, parameter, coef_max, coef_min)
    datad.append(datad0)
    for i in range(6):
        datad.append(datad2[i])
    pred = []
    for i in range(7):
        pred.append(float(dataneed[2]) + datad[i])
    return pred


def get_normalization_info(data):
    avg = data.mean()
    std = data.std()
    return avg, std
