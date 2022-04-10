import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import buildModel as bM
import transModel as tM
import modelEval as mE
import mergeResult as mR
import DRLstm
import pandas as pd
from tensorflow import keras
import datetime
import yaml
import json
import glob
import numpy as np
from matplotlib import pyplot as plt

'''
1.归一化：source、trans-预测、训练和迁移都用厦门湾；target-三沙湾
2.模型测试：增加纯、混差分模型比较
3.纯、混差分模型系数：都用三沙湾系数（通过遍历得到，步长为0.005）
'''


class TestTransfer:

    def __init__(self, param,
                 source_data, target_data,
                 source_name, target_name,
                 split_rate,
                 coef_dic_source,
                 coef_dic_target,
                 x_step=27, y_step=7,
                 retrain=None,
                 run_id=None):
        """
        模型迁移测试类。
        coef_dic,包含pure_coef_max, mix_coef_max, pure_coef_min, mix_coef_min  4个值为列表的键
        eg. TransTest(xm1['max'], ypj1['max'], 'xm1', 'ypj1', 0.3)
        :param source_data: 源域数据(Series)
        :param target_data: 目标域训练数据(Series)
        :param source_name: 源域站点名称(str)
        :param target_name: 目标域站点名称(str)
        :param split_rate: 用于训练目标域模型的数据比例（应设置的比较小，对应在数据量较小的地区进行训练的研究思路）
        """
        self.param = param
        self.source_data = source_data  # 源域数据
        self.target_data = target_data[:int(len(target_data) * split_rate)]  # 目标域训练数据（用于训练模型）
        assert self.source_data.name == self.param
        assert self.target_data.name == self.param
        self.coef_dic_source = coef_dic_source  # 保存在info文件中，modelEval模块通过info文件读取
        self.coef_dic_target = coef_dic_target

        self.source_avg, self.source_std = DRLstm.get_normalization_info(self.source_data)
        self.target_avg, self.target_std = DRLstm.get_normalization_info(self.target_data)

        self.test_data = target_data[int(len(target_data) * split_rate):]  # 目标域测试数据

        self.source_name = source_name  # 源域站点名称
        self.target_name = target_name  # 目标域站点名称

        self.x_step = x_step
        self.y_step = y_step

        if retrain is None:
            self.retrain = []
        else:
            self.retrain = retrain  # 是否重新训练并覆盖原模型

        self.source_model = None  # 源域模型
        self.target_model = None  # 目标域模型
        self.trans_model = None  # 目标域迁移模型

        self.bm_source = None
        self.bm_target = None
        self.tm = None

        self.split_rate = split_rate

        if run_id is None:
            self.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M')  # 生成run目录
        else:
            self.run_id = run_id

    def build_source_model(self):
        self.bm_source = bM.BuildModel(self.source_data, self.source_name,
                                       self.source_avg, self.source_std,
                                       self.x_step, self.y_step)
        self.bm_source.run()

    def build_target_model(self):
        self.bm_target = bM.BuildModel(self.target_data, self.target_name,
                                       self.target_avg, self.target_std,
                                       self.x_step, self.y_step)
        self.bm_target.run()

    def get_source_model(self):
        source_model_path = '.\\model\\base\\%s.h5' % self.source_name
        if not os.path.exists(source_model_path):
            print("文件名为| %s |的源域模型不存在，开始训练模型" % source_model_path)
            self.build_source_model()
        else:
            if 'source' in self.retrain:
                print("文件名为| %s |的源域模型存在，重新训练" % source_model_path)
                self.build_source_model()
            else:
                print("文件名为| %s |的源域模型存在" % source_model_path)
        self.source_model = keras.models.load_model(source_model_path)

    def get_target_model(self):
        target_model_path = '.\\model\\base\\%s.h5' % self.target_name
        if not os.path.exists(target_model_path):
            print("文件名为| %s |的目标域模型不存在，开始训练模型" % target_model_path)
            self.build_target_model()
        else:
            if 'target' in self.retrain:
                print("文件名为| %s |的目标域模型存在，重新训练" % target_model_path)
                self.build_target_model()
            else:
                print("文件名为| %s |的目标域模型存在" % target_model_path)
        self.target_model = keras.models.load_model(target_model_path)

    def build_trans_model(self):
        if self.source_model is None:
            self.get_source_model()
        self.tm = tM.TransModel(self.source_model, self.target_data, "%s_to_%s" % (self.source_name, self.target_name),
                                # self.target_avg, self.target_std,  # lxq迁移模型时用目标域数据标准化
                                self.source_avg, self.source_std,  # lxq迁移模型时用源域数据标准化
                                self.x_step, self.y_step)  # 迁移模型时用目标域数据
        self.tm.run()

    def get_trans_model(self):
        trans_model_path = '.\\model\\trans\\%s_to_%s.h5' % (self.source_name, self.target_name)
        if not os.path.exists(trans_model_path):
            print("文件名为| %s |的迁移模型不存在，开始迁移模型" % trans_model_path)
            self.build_trans_model()
        else:
            if 'trans' in self.retrain:
                print("文件名为| %s |的迁移模型存在，重新迁移" % trans_model_path)
                self.build_trans_model()
            else:
                print("文件名为| %s |的迁移模型存在" % trans_model_path)
        self.trans_model = keras.models.load_model(trans_model_path)

    def run(self):
        # lxq各自训练n个模型挑选出最优，还是在1个run中随机各创建1个模型直接比较？
        run_path = '.\\test\\run_%s' % self.run_id
        if not os.path.exists(run_path):
            os.mkdir(run_path)

        info = {'param': self.param,
                'source_name': self.source_name,
                'target_name': self.target_name,
                'split_rate': self.split_rate,
                'x_step': self.x_step,
                'y_step': self.y_step,
                'source_avg': float(self.source_avg),
                'source_std': float(self.source_std),
                'target_avg': float(self.target_avg),
                'target_std': float(self.target_std),
                'coef_dic_source': self.coef_dic_source,
                'coef_dic_target': self.coef_dic_target,
                }

        with open('%s\\info.yaml' % run_path, 'w') as f:  # 保存配置文件
            yaml.dump(info, f)

        self.source_data.to_csv('%s\\source_data.csv' % run_path)
        self.target_data.to_csv('%s\\target_data.csv' % run_path)
        self.test_data.to_csv('%s\\test_data.csv' % run_path)

        self.get_source_model()
        self.source_model.save('%s\\source_model.h5' % run_path)
        self.get_target_model()
        self.target_model.save('%s\\target_model.h5' % run_path)
        self.get_trans_model()
        self.trans_model.save('%s\\trans_model.h5' % run_path)
        # plt.plot(self.tm.history.history['val_loss'], label='val_loss')
        # plt.plot(self.tm.history.history['loss'], label='loss')
        plt.legend()
        plt.savefig('%s\\loss.png' % run_path)
        plt.close()

        mt = mE.ModelTest(self.run_id, self.x_step, self.y_step)
        mt.create_test_dataset()
        mt.model_pred()
        rmse, r2, mae, mape, acc = mt.eval_model()
        rmse.to_csv('%s\\rmse.csv' % run_path)
        r2.to_csv('%s\\r2.csv' % run_path)
        mae.to_csv('%s\\mae.csv' % run_path)
        mape.to_csv('%s\\mape.csv' % run_path)
        acc.to_csv('%s\\acc.csv' % run_path)

        json_str = self.source_model.to_json()
        json_dic = json.loads(json_str)
        with open('%s\\model_struct.json' % run_path, 'w') as f:
            json.dump(json_dic, f, indent=4)


class Runset:

    def __init__(self, name):
        self.name = name  # 测试集合名称

    def do_test(self, num,
                param, source_data, target_data, source_name, target_name, split_rate,
                coef_dic_source, coef_dic_target):
        for test_num in range(num):
            tt = TestTransfer(param, source_data, target_data,
                              source_name, target_name, split_rate,
                              coef_dic_source, coef_dic_target,
                              retrain=['source', 'target', 'trans'],
                              run_id='%s_%s' % (self.name, test_num))
            tt.run()

    def collate_test(self):
        mR.create_run_set(self.name)
        for metric in ['rmse', 'r2', 'acc', 'mae', 'mape']:
            mR.settle_run_set(self.name, metric)
        del_list = glob.glob('.\\test\\runset_%s\\run_*' % self.name)
        for file in del_list:
            if os.path.isdir(file):
                mR.del_file(file)


if __name__ == '__main__':
    xm1 = pd.read_csv('.\\data\\xm_1.csv', index_col=0, parse_dates=True)  # 读取厦门湾数据
    xm2 = pd.read_csv('.\\data\\xm_2.csv', index_col=0, parse_dates=True)
    xm3 = pd.read_csv('.\\data\\xm_3.csv', index_col=0, parse_dates=True)
    ypj1 = pd.read_csv('.\\data\\ypj1.csv', index_col=0, parse_dates=True)  # 读取渔排基1号数据
    ypj2 = pd.read_csv('.\\data\\ypj2.csv', index_col=0, parse_dates=True)  # 读取渔排基2号数据
    ypj3 = pd.read_csv('.\\data\\ypj3.csv', index_col=0, parse_dates=True)  # 读取渔排基3号数据
    jy = pd.read_csv('.\\data\\jy.csv', index_col=0, parse_dates=True)

    coef_dic_target_ypj_test = {
        'pure_coef_max': [0.02651999, 0.03266589, 0.02327083, [0.01584, 0.02436, 0.04041, 0.06448, 0.10615, 0.1386]],
        'mix_coef_max': [0.72648581, 0.02510776, [0.00278, 0.00704, 0.02157, 0.05391, 0.09833, 0.12848]],
        'pure_coef_min': [0.02651999, 0.03266589, 0.02327083, [0.01584, 0.02436, 0.04041, 0.06448, 0.10615, 0.1386]],
        'mix_coef_min': [0.72648581, 0.02510776, [0.00278, 0.00704, 0.02157, 0.05391, 0.09833, 0.12848]]
    }

    coef_dic_source_ypj_test = {
        'pure_coef_max': [0.02651999, 0.03266589, 0.02327083, [0.01584, 0.02436, 0.04041, 0.06448, 0.10615, 0.1386]],
        'mix_coef_max': [0.72648581, 0.02510776, [0.00278, 0.00704, 0.02157, 0.05391, 0.09833, 0.12848]],
        'pure_coef_min': [0.02651999, 0.03266589, 0.02327083, [0.01584, 0.02436, 0.04041, 0.06448, 0.10615, 0.1386]],
        'mix_coef_min': [0.72648581, 0.02510776, [0.00278, 0.00704, 0.02157, 0.05391, 0.09833, 0.12848]]
    }

    test_source = {'xm1': xm1, 'xm2': xm2, 'xm3': xm3}
    test_target = {'ypj1': ypj1, 'ypj2': ypj2, 'ypj3': ypj3}
    test_param = ['tem_max', 'tem_min']
    for sk, sv in test_source.items():
        for tk, tv in test_target.items():
            for param in test_param:
                r = Runset('(%s)(%s)_%s' % (sk, tk, param))
                r.do_test(100, param, sv[param], tv[param], '%s_%s' % (sk, param), '%s_%s' % (tk, param), 0.4,
                          coef_dic_target_ypj_test, coef_dic_source_ypj_test)
                r.collate_test()
