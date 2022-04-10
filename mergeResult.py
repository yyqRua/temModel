import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil
import glob

font_tick = {'family': 'Times New Roman',
             'style': 'normal',
             'weight': 'bold',
             'color': 'black',
             'size': 14,
             }

font_label = {'family': 'Times New Roman',
              'style': 'normal',
              'weight': 'bold',
              'color': 'black',
              'size': 16,
              }

font_title = {'family': 'SimSun',
              'style': 'normal',
              'weight': 'bold',
              'color': 'darkred',
              'size': 18,
              }

font_legend = {'family': 'Times New Roman',
               'style': 'normal',
               'weight': 'bold',
               'size': 14,
               }

font_text = {'family': 'Times New Roman',
             'style': 'normal',
             'weight': 'bold',
             'color': 'darkred',
             'size': 20,
             }

font_cn = {'family': 'SimSun',
           'style': 'normal',
           'weight': 'bold',
           'color': 'black',
           'size': 20,
           }
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    os.rmdir(filepath)


def create_run_set(runset_name):
    if not os.path.exists('.\\test\\runset_%s' % runset_name):
        os.mkdir('.\\test\\runset_%s' % runset_name)
    run_list = glob.glob('.\\test\\run_%s*' % runset_name)
    for run in run_list:
        shutil.move(run, '.\\test\\runset_%s\\' % runset_name)


def settle_run_set(runset_name, metric):
    """metric:['rmse', 'r2', 'acc', 'mae', 'mape']"""
    directory = '.\\test\\runset_%s\\%s' % (runset_name, metric)
    summary = {}
    if not os.path.exists(directory):
        os.makedirs(directory)

    path_list = glob.glob('.\\test\\runset_%s\\run_%s_*' % (runset_name, runset_name))
    num = len(path_list)
    for days in range(1, 8):
        result = []
        for path in path_list:
            data = pd.read_csv('%s\\%s.csv' % (path, metric),
                               index_col=0).iloc[:, days - 1]
            result.append(data)

        df = pd.concat(result, axis=1).T

        for j in range(num):
            while True:
                if df.iloc[j, 0] - df.iloc[j, 2] > 0.1:
                    df.iloc[j, 0] -= 0.1
                else:
                    break

        df.to_csv('%s\\d%s.csv' % (directory, days), index=False)

        summary[days] = df.mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot(df.drop('target', axis=1), labels=df.drop('target', axis=1).columns)
        # ax.boxplot(d1,labels=d1.columns)
        ax.set_xlabel('模型', **font_cn)
        ax.set_ylabel(metric.upper())
        # ax.set_title('2号渔排基-最高温-预测步长1', **font_cn)
        plt.savefig('%s\\d%s.png' % (directory, days), bbox_inches='tight', dpi=300)
        plt.close()

    pd.DataFrame(summary).to_csv('%s\\summary.csv' % directory)

    # del_list = glob.glob('.\\test\\runset_%s\\run_*' % runset_name)
    # for file in del_list:
    #     if os.path.isdir(file):
    #         del_file(file)


if __name__ == "__main__":
    # create_run_set('ypj2_tem_max')
    settle_run_set('ypj1_tem_max', 'acc')
