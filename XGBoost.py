import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier

import const

warnings.filterwarnings('ignore')
Label = 'label'


class Example_XGB:
    def __init__(self, filePath, cols):
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # 读入表格文件函数
        self.all = pd.read_csv(filePath, encoding='UTF-8')
        own_feature = self.all.columns.values  # 数据集具备的特征，包含标签label
        self.feature_cols = self.get_feature(cols, own_feature)  # cols：需要的特征，feature_cols: 特征交集
        # if self.feature_cols == "err":
        #     err = "err&模型加载错误或测试文件读取失败！"
        self.y_pred = []

        self.model = XGBClassifier()
        print("初始化完成...")

    def split_file(self, test_file_path, train_file_path):
        if len(self.feature_cols) == 0:
            return "err"
        X = self.all
        y = X.pop(Label)  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.3)

        X_test = X_test[self.feature_cols]
        X_train = X_train[self.feature_cols]

        train_data = pd.concat([X_train, Y_train], axis=1)
        train_data.to_csv(train_file_path, index=False, encoding='UTF-8', mode='w')  # 训练集文件
        test_data = pd.concat([X_test, Y_test], axis=1)
        test_data.to_csv(test_file_path, index=False, encoding='UTF-8', mode='w')  # 测试集文件

        return ""

    # 特征处理
    def get_feature(self, feature1, feature2):
        # 两种求列表并集的方法
        feature3 = (set(feature1) - set(feature2))  # 需要但是不具备的特征
        feature4 = list(set(feature1) - feature3)  # 需要且具备的特征
        if len(feature4) <= 1:
            return "err"
        else:
            return feature4

        # 传 树形图 ，柱状图
        # filePath 文件路径
        # feature_cols 标签

    def process(self, train, path1, path2):
        # 数据处理，模型调用
        # self.plot_feature_importance(train, path2)
        fmap_filename = "picture/xgb_2.fmap"
        self.tree_pic(self.feature_cols, fmap_filename, path1)
        self.plot_feature_importance(train, path2)

    def train_model(self, train_file, model_file):
        train_data = pd.read_csv(train_file)
        x_train = train_data
        y_train = x_train.pop(Label)
        self.model.fit(x_train, y_train)
        if not os.path.exists(model_file):
            f = open(model_file, mode='ab')
        else:
            f = open(model_file, mode="wb")
        pickle.dump(self.model, f)  # 保存模型

        return x_train

    def result(self, all_file, model_file, test_file_path, img_path):
        if model_file != "" or all_file == "" or test_file_path == "":
            f = open(model_file, 'rb')
            self.model = pickle.load(f)  # 读取模型
        else:
            return "err"

        train_data = pd.read_csv(test_file_path)
        x_test = train_data
        y_test = x_test.pop(Label)

        y_pred = self.model.predict(x_test)  # 模型测试

        # 计算评价指标
        accuracy = self.model.score(x_test, y_test)
        accuracy = '%.4f%%' % (accuracy * 100)
        ret = "准确率:{0}".format(accuracy)
        print("ret:", ret)

        y_pred = y_pred.reshape(y_pred.shape[0], 1)
        res = pd.read_csv(test_file_path, encoding='utf-8')
        # a = pd.read_csv(all_file, encoding='utf-8')
        # res = a.loc[y_test.index]
        res['pred'] = y_pred
        res = res.iloc[0:500]
        res.to_csv(img_path, mode='w', index=False, encoding='UTF-8')
        return ret

    def tree_pic(self, features, fmap_filename, path_1):
        outfile = open(fmap_filename, 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        from xgboost import plot_tree
        plot_tree(self.model, num_trees=0, fmap=fmap_filename)
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        fig.savefig(path_1)
        # im = Image.open(path_1)
        # im.show()

    def plot_feature_importance(self, x_train, path2):
        plt.clf()  # 清空画板
        feat_labels = x_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(x_train.shape[1]):
            print("%2d)  %-*s  %f" % (f + 1, 30,
                                      feat_labels[f],
                                      importances[indices[f]]))
        plt.title('特征重要性分析', fontsize=18)
        plt.bar(range(x_train.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')
        font2 = {'size': 18}
        plt.xlabel(u'特征变量', font2)
        plt.ylabel(u'重要度', font2)
        plt.xticks(range(x_train.shape[1]),
                   feat_labels, rotation=0, fontsize=16)
        plt.yticks(fontsize=18)
        plt.xlim([-1, x_train.shape[1]])
        plt.tight_layout()
        plt.savefig(path2)
        # fig = plt.gcf()
        # fig.savefig(path2)
        # plt.show()

    # XGBoost结果可视化


def deal_real(data):
    name = data.get("name", "")
    if name != "":
        seq = data.get("seq", "")
        fea = get_featureby_by_seq(seq)
        all_file = "{}/{}".format("data", data.get("all_file", ""))
        step = data.get("step", "")
        model_file = const.model_file_format.format(name)
        train_file = const.train_file_format.format(const.CURRENT_DIR, name)
        test_file = const.test_file_format.format(const.CURRENT_DIR, name)
        print("name:", name, "fea:", fea, "all_file:", all_file, "step:", step)
        if step == const.SPLIT:
            print("开始划分数据...")
            return split_file(name, all_file, test_file, train_file, fea)
        elif step == const.TRAIN:
            print("开始训练数据...")
            return train_model(name, all_file, fea, train_file, model_file)
        elif step == const.TEST:
            print("开始测试数据...")
            return test_mdoel(name, all_file, fea, test_file, model_file)
        else:
            return const.err_format.format("step error")

    return const.err_format.format("name must have")


def get_featureby_by_seq(seq):
    feature = []
    if seq == "a":
        feature = ['TOA', 'PA', 'RF_START', 'DOA']  # 准确率高的
    elif seq == "b":
        feature = ['Beam', 'PA', 'PAL', 'PA_MID']  # 准确率较低的
    elif seq == "c":
        feature = ['PAL', 'PAR', 'PA3']  # 准确低的
    return feature


def split_file(name, all_file, train_file, test_file, feature):
    ex = Example_XGB(all_file, feature)
    err = ex.split_file(test_file, train_file)
    if err == "err":
        return const.err_format.format("特征不足，无法分选")
    else:
        ret = "{}&{}${}${}".format(name, const.SPLIT, train_file, test_file)
        return ret


def train_model(name, all_file, feature, train_file, model_file):
    ex = Example_XGB(all_file, feature)
    train = ex.train_model(train_file, model_file)
    path_1 = const.img_tree_format.format(const.CURRENT_DIR, name)
    path_2 = const.img_impr_format.format(const.CURRENT_DIR, name)
    ex.process(train, path_1, path_2)

    ret = "{}&{}${}#{}".format(name, const.TRAIN, path_1, path_2)
    return ret


def test_mdoel(name, all_file, feature, test_file, model_file):
    ex = Example_XGB(all_file, feature)
    img_path = const.result_file_format.format(const.CURRENT_DIR, name)

    ret = ex.result(all_file, model_file, test_file, img_path)
    ret = "{}&{}${}#{}".format(name, const.TEST, ret, img_path)
    return ret


if __name__ == '__main__':
    print()
