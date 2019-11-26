import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image
from matplotlib import pyplot as plt
from pylab import mpl
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import const

Label = 'label'


def get_feature(feature1, feature2):
    # 两种求列表并集的方法
    print("----", feature1)
    print('/////', feature2)
    ret = []
    for i in feature2:
        if i in feature1:
            ret.append(i)
    print(ret)
    if len(ret) <= 1:
        return "err"
    else:
        return ret


class Example:
    def __init__(self, name, filepath, cols):
        self.check_file()
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        warnings.filterwarnings('ignore')

        print(filepath)
        self.all = pd.read_csv(filepath, encoding='UTF-8')

        own_feature = self.all.columns.values  # 数据集具备的特征，包含标签class
        self.feature_cols = get_feature(cols, own_feature)  # cols：需要的特征，feature_cols: 特征交集

        if name == const.RANDOM_FOREST:
            self.model = RandomForestClassifier()
        elif name == const.DECISION:
            self.model = DecisionTreeClassifier()  # 所有参数均置为默认状态 # 初始化模型

    def check_file(self):
        data_path = "data"
        test_path = "data/data_extraction"
        model_path = "model"
        pic_path = "picture"  # 图片文件
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    # 划分数据
    def split_file(self, test_file_path, train_file_path):
        if len(self.feature_cols) <= 1:
            return "err"

        X = self.all
        y = X.pop(Label)  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.2)
        # 求出特征交集

        X_test = X_test[self.feature_cols]
        X_train = X_train[self.feature_cols]

        train_data = pd.concat([X_train, Y_train], axis=1)
        train_data.to_csv(train_file_path, index=False, encoding='UTF-8', mode='w')  # 训练集文件

        test_data = pd.concat([X_test, Y_test], axis=1)
        test_data.to_csv(test_file_path, index=False, encoding='UTF-8', mode='w')  # 测试集文件

        return ""

    # 特征处理

    def train_model(self, train_file, model_file):
        train_data = pd.read_csv(train_file)
        x_train = train_data
        y_train = x_train.pop(Label)

        if os.path.exists(model_file):
            f = open(model_file, mode="rb")  # 只读模式
            self.model = pickle.load(f)
        self.model.fit(x_train, y_train)
        if not os.path.exists(model_file):
            f = open(model_file, mode='ab')  # 追加模式
        else:
            f = open(model_file, mode="wb")  # 写入模式
        pickle.dump(self.model, f)  # 保存模型
        return x_train

    def process(self, name, x_train, path1, path2):
        if name == const.RANDOM_FOREST:
            self.tree_pic_for_random(path1)
        elif name == const.DECISION:
            self.tree_pic(path1)
        self.plot_feature_importance(x_train, path2)

    def tree_pic(self, img_path):
        dot_tree = tree.export_graphviz(self.model, out_file=None, feature_names=self.feature_cols, max_depth=5,
                                        filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_tree)
        Image(graph.create_png())
        graph.write_png(img_path)

    def tree_pic_for_random(self, path1):
        from sklearn.tree import export_graphviz
        estimator = self.model.estimators_[5]
        export_graphviz(estimator, out_file='tree.dot', max_depth=4,
                        feature_names=self.feature_cols,
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        # 用系统命令转为PNG文件(需要 Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', path1, '-Gdpi=600'])

    def plot_feature_importance(self, x_train, path2):
        plt.clf()  # 清空画板
        feat_labels = x_train.columns
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        for f in range(x_train.shape[1]):
            print("%2d)  %-*s  %f" % (f + 1, 30,
                                      feat_labels[f],
                                      importance[indices[f]]))
        plt.title('变量重要性分析', fontsize=18)
        plt.bar(range(x_train.shape[1]),
                importance[indices],
                color='lightblue',
                align='center')
        # ax.set_xlabel(..., fontsize=20)
        font2 = {'size': 18}
        plt.xlabel(u'特征变量', font2)
        plt.ylabel(u'重要度', font2)
        plt.xticks(range(x_train.shape[1]),
                   feat_labels, rotation=0, fontsize=16)
        plt.yticks(fontsize=18)
        plt.xlim([-1, x_train.shape[1]])
        plt.tight_layout()
        plt.savefig(path2)
        # plt.show()

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
        res.to_csv(img_path, index=False, encoding='UTF-8', mode='w')
        return ret


def deal(data):
    print("data:", data)
    name = data.get("name", "")
    if name != "":
        seq = data.get("seq", "")
        fea = get_featureby_by_name(name, seq)
        all_file = "{}/{}".format("data", data.get("all_file", ""))
        step = data.get("step", "")
        model_file = const.model_file_format.format(name)
        train_file = const.train_file_format.format(const.CURRENT_DIR, name)
        test_file = const.test_file_format.format(const.CURRENT_DIR, name)
        if step == const.SPLIT:
            return split_file(name, all_file, test_file, train_file, fea)
        elif step == const.TRAIN:
            return train_model(name, all_file, fea, train_file, model_file)
        elif step == const.TEST:
            return test_mdoel(name, all_file, fea, test_file, model_file)
        else:
            return const.err_format.format("step must have")

    print("name:", name)
    return const.err_format.format("无算法名称")


def get_featureby_by_name(name, seq):
    feature = []
    if name == const.DECISION:
        feature = ['TOA', 'PA', 'PW', 'DOA']
    elif name == const.RANDOM_FOREST:
        if seq == "a":
            feature = ['TOA', 'TOE', 'PA', 'RF_START', 'RF_MID', 'RF_END', 'DOA']  # 准确率高的
        elif seq == "b":
            feature = ['Beam', 'PA_MID']  # 准确率较低的
        elif seq == "c":
            feature = ['PA_MID', 'DOA', 'PA3', '仰角']  # 准确低的
    return feature


def split_file(name, all_file, test_file, train_file, feature):
    ex = Example(name, all_file, feature)
    err = ex.split_file(test_file, train_file)
    if err == "err":
        return const.err_format.format("特征不足，无法分选")
    ret = "{}&{}${}${}".format(name, const.SPLIT, train_file, test_file)
    return ret


def train_model(name, all_file, feature, train_file, model_file):
    ex = Example(name, all_file, feature)
    train = ex.train_model(train_file, model_file)
    tree_img = const.img_tree_format.format(const.CURRENT_DIR, name)
    impr_img = const.img_impr_format.format(const.CURRENT_DIR, name)
    ex.process(name, train, tree_img, impr_img)

    ret = "{}&{}${}#{}".format(name, const.TRAIN, tree_img, impr_img)
    return ret


def test_mdoel(name, all_file, feature, test_file, model_file):
    ex = Example(name, all_file, feature)
    ret_file = const.result_file_format.format(const.CURRENT_DIR, name)
    ret = ex.result(all_file, model_file, test_file, ret_file)
    if ret == "err":
        return const.err_format.format("模型加载错误或测试文件读取失败！")

    ret = "{}&{}${}#{}".format(name, const.TEST, ret, ret_file)
    return ret


if __name__ == '__main__':
    data = {
        "name": const.RANDOM_FOREST,
        "seq": "a",
        "all_file": "data/random_PDW1.csv",
        "test_file": "data/data_extraction/random_forest_test.csv",
        "train_file": "data/data_extraction/random_forest_train.csv",
        "step": "test",
    }
    ret = deal(data)
    print(ret)
