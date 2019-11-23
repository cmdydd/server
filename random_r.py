# -*- coding: utf-8 -*-
# 1.
# 名称：random__tree_1.png
# 路径： path1 = "/home/ubuntu/PycharmProjects/web_server/picture/random__tree_1.png"

# 2.特征重要性图，显示在实验过程中
# 名称：random_variable_importance1.png
# 路径：path2 = "/home/ubuntu/PycharmProjects/web_server/picture/random_variable_importance_1.png"
# 目前存在的问题
# 1.随机森林的特征重要性图为空
# 解决方案：将随机森林的特征重要性path2写死，在path2路径下保存特征重要性图，修改生成的特征重要性图的路径为path4

# 3.分选准确率，显示在实验结果中
# Accuracy:70%


# 4.csv文件“预测结果”，显示在实验结果中，可以像上传数据那样导入数据。
# 路径：res.to_csv(r"/home/ubuntu/PycharmProjects/web_server/picture/random_predict_result_1.csv",
#            mode='a', index=False)
# 路径：path3="/home/ubuntu/PycharmProjects/web_server/picture/random_predict_result_1.csv"

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

import const


class Example:
    def __init__(self, filePath, cols):
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        warnings.filterwarnings('ignore')

        self.train = pd.read_csv(filePath, encoding='UTF-8')
        X = self.train
        y = X.pop('label')  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        self.feature_cols = cols
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(X, y, test_size=0.2)
        self.X_test = self.X_test[self.feature_cols]
        self.X_train = self.X_train[self.feature_cols]
        self.y_pred = []

        self.model = RandomForestClassifier()
        print(  # 数据介绍
            '输入数据：7种雷达辐射源，每类辐射源均有23维特征'"\n"
            '-------------------------------------------------------------------------------------------------'"\n"
            'TOA: 脉冲前沿时间（到达时间）'"\n"
            'TOE: 脉冲后沿时间'"\n"
            'PA: 幅度'"\n"
            'PW: 脉宽（TOE-TOA）'"\n"
            'RF_START: 脉冲测量的频率参数'"\n"
            'RF_MID: '"\n"
            'RF_END: '"\n"
            'Beam: 波束，脉冲检测在不同波束的幅度不同'"\n"
            'CH_NUM: 射频通道'"\n"
            'BOX_NUM: 算法中间的一个运算量'"\n"
            'Frame_NUM: 帧号'"\n"
            'PA_MID: 脉宽中间幅度'"\n"
            'PAL: 6个波束采到，主波束左边值'"\n"
            'PAR: 6个波束采到，主波束右边值'"\n"
            'PA3: 波束采集参数'"\n"
            'PAP	               '"\n"
            'PA_NY: 逆影'"\n"
            'DOA: 方位角'"\n"
            '仰角: 辐射源与接收机所在处的地平线之间的夹角'"\n"
            'PA_diff: 主波束-逆影'"\n"
            'Flag: 硬件控制表号'"\n"
            '源雷控序列号: 硬件控制表中参数'"\n"
            '尾部中间频率: 倒数第6个点频率，RF_END前6个点'"\n"
            '-------------------------------------------------------------------------------------------------'

        )

    # 传 树形图 ，柱状图
    # filePath 文件路径
    # feature_cols 标签
    def process(self, path1, path2):
        # 数据处理，模型调用
        self.model.fit(self.X_train, self.Y_train)
        self.plot_feature_importance(path2)
        self.tree_pic(path1)

    # 生成一棵树的图像
    # 树形文件路径
    def tree_pic(self, path1):
        from sklearn.tree import export_graphviz
        estimator = self.model.estimators_[5]
        export_graphviz(estimator, out_file='tree.dot', max_depth=4,
                        feature_names=self.feature_cols,
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        # 用系统命令转为PNG文件(需要 Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', path1, '-Gdpi=600'])

    def plot_feature_importance(self, path2):
        feat_labels = self.X_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(self.X_train.shape[1]):
            print("%2d)  %-*s  %f" % (f + 1, 30,
                                      feat_labels[f],
                                      importances[indices[f]]))
        plt.title('特征重要性分析', fontsize=15)
        font2 = {'size': 20}
        plt.xlabel(u'特征变量', font2)
        plt.ylabel(u'重要度', font2)
        plt.bar(range(self.X_train.shape[1]),
                importances[indices],
                color='lightblue',
                align='center', width=0.6)
        plt.xticks(range(self.X_train.shape[1]),
                   feat_labels, rotation=0, fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(path2)
        plt.show()

    def result(self, path3):
        # 计算评价指标
        self.y_pred = self.model.predict(self.X_test)
        accuracy = self.model.score(self.X_test, self.Y_test)
        accuracy = '%.4f%%' % (accuracy * 100)
        recall = metrics.recall_score(self.Y_test, self.y_pred, average='macro')
        recall = '%.4f%%' % (recall * 100)
        f1 = metrics.f1_score(self.Y_test, self.y_pred, average='weighted')
        f1 = '%.4f%%' % (f1 * 100)
        ret = "准确率:{0}\n召回率:{1}\nF-1 score:{2}".format(accuracy, recall, f1)
        # ret = "\033[1;31;40m准确率:{:.2%}\033[0m\nRecall:{}\nF-1 score:{}".format(accuracy, recall, f1)

        # 输出预测值与原有分类的对比
        self.y_pred = self.y_pred.reshape(self.y_pred.shape[0], 1)
        # a = pd.read_csv(filePath, encoding='gbk')
        res = self.train.loc[self.Y_test.index]
        res['pred'] = self.y_pred
        res = res.iloc[0:500]
        res.to_csv(path3, mode='a',
                   index=False)
        return ret


def d(fea, path):
    # path = 'data/random_PDW1.csv'
    path = "data/{}".format(path)
    if fea == "a":
        feature = ['TOA', 'TOE', 'PA', 'RF_START', 'RF_MID', 'RF_END', 'DOA']  # 准确率高的
    elif fea == "b":
        feature = ['Beam', 'PA_MID']  # 准确率较低的
    elif fea == "c":
        feature = ['PA_MID', 'DOA', 'PA3', '仰角']  # 准确低的
    else:
        print("特征传输错误")
        return "error"
    ex = Example(path, feature)

    path1 = "picture/random_tree.png"
    path2 = "picture/random_variable_importance.png"
    path3 = "picture/random_predict_result.csv"
    ex.process(path1, path2)

    ret = ex.result(path3)
    path1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path1)
    path2 = const.DIR_FORMAT.format(const.CURRENT_DIR, path2)
    path3 = const.DIR_FORMAT.format(const.CURRENT_DIR, path3)

    if fea == 'a':
        path2 = "picture/random_variable_importance_a.png"
    elif fea == 'b':
        path2 = "picture/random_variable_importance_b.png"
    elif fea == 'c':
        path2 = "picture/random_variable_importance_c.png"

    path2 = const.DIR_FORMAT.format(const.CURRENT_DIR, path2)
    str = "randomforest&{}#{}${}#{}".format(path1, path2, ret, path3)
    return str


if __name__ == '__main__':
    print("main")
    fea = 'c'
    path = 'random_PDW1.csv'
    d(fea, path)
