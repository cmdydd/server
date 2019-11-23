import pickle
import warnings

import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image
from matplotlib import pyplot as plt
from pylab import mpl
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import const


class Example:
    def __init__(self, filepath, cols):
        self.ss = None
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        warnings.filterwarnings('ignore')

        self.train = pd.read_csv(filepath, encoding='UTF-8')
        own_feature = self.train.columns.values  # 数据集具备的特征，包含标签class
        self.feature_cols = self.get_feature(cols, own_feature)  # cols：需要的特征，feature_cols: 特征交集

        X = self.train
        y = X.pop('label')  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(X, y, test_size=0.2)
        # 求出特征交集

        self.X_test = self.X_test[self.feature_cols]
        self.X_train = self.X_train[self.feature_cols]
        self.y_pred = []
        self.model = DecisionTreeClassifier()  # 所以参数均置为默认状态
        # print("model:", self.model)
        # print("tree:", tree)

    # 文件导出
    def file_out(self, path4, path5):
        train_data = pd.concat([self.X_train, self.Y_train], axis=1)
        train_file = train_data.to_csv(path4, encoding='UTF-8')  # 训练集文件
        test_data = pd.concat([self.X_test, self.Y_test], axis=1)
        test_file = test_data.to_csv(path5, encoding='UTF-8')  # 测试集文件
        return path4, path5

    # 特征处理
    def get_feature(self, feature1, feature2):
        # 两种求列表并集的方法
        feature3 = (set(feature1) - set(feature2))  # 需要但是不具备的特征
        feature4 = list(set(feature1) - feature3)  # 需要且具备的特征
        if len(feature4) <= 0:
            print("数据特征不足，无法分选")
            return
        else:
            return feature4

    def process(self, path1, path2):
        self.X_train = self.X_train[self.feature_cols]
        self.X_test = self.X_test[self.feature_cols]
        self.model.fit(self.X_train, self.Y_train)
        self.tree_pic(path1)
        self.plot_feature_importance(path2)

    def tree_pic(self, img_path):
        print("self.model:", self.model)
        dot_tree = tree.export_graphviz(self.model, out_file=None, feature_names=self.feature_cols, max_depth=5,
                                        filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_tree)
        Image(graph.create_png())
        graph.write_png(img_path)

    def plot_feature_importance(self, path2):
        feat_labels = self.X_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(self.X_train.shape[1]):
            print("%2d)  %-*s  %f" % (f + 1, 30,
                                      feat_labels[f],
                                      importances[indices[f]]))
        plt.title('变量重要性分析', fontsize=20)
        plt.bar(range(self.X_train.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')
        # ax.set_xlabel(..., fontsize=20)
        font2 = {'size': 20}
        plt.xlabel(u'特征变量', font2)
        plt.ylabel(u'重要度', font2)
        plt.xticks(range(self.X_train.shape[1]),
                   feat_labels, rotation=0, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(path2)
        plt.show()

    def result(self, img_path):
        clf2 = pickle.loads(self.ss)  # 读取模型
        y_pred = clf2.predict(self.X_test)  # 模型测试
        # y_pred = self.model.predict(self.X_test)

        # 计算评价指标
        accuracy = self.model.score(self.X_test, self.Y_test)
        accuracy = '%.4f%%' % (accuracy * 100)
        recall = metrics.recall_score(self.Y_test, y_pred, average='macro')
        recall = '%.4f%%' % (recall * 100)
        f1 = metrics.f1_score(self.Y_test, y_pred, average='weighted')
        f1 = '%.4f%%' % (f1 * 100)
        ret = "准确率:{0}\n召回率:{1}\nF-1 score:{2}".format(accuracy, recall, f1)
        print("ret:", ret)

        y_pred = y_pred.reshape(y_pred.shape[0], 1)
        a = pd.read_csv('data/decision_PDW.csv', encoding='gbk')
        res = a.loc[self.Y_test.index]
        res['pred'] = y_pred
        res.to_csv(img_path, mode='a', index=False)

        return ret


def deal(path):
    print("path:", path)
    path = "data/{}".format(path)
    # feature = ['TOA', 'PA', 'PW', 'F', 'DOA', 'wa']  # 需要的特征多
    feature = ['TOA', 'PA', 'PW', 'F', 'DOA']
    # feature = ['TOA', 'PA', 'F', 'DOA'] # 需要的特征多

    ex = Example(path, feature)

    path4 = "data/data_extraction/decision_train.csv"
    path5 = "data/data_extraction/decision_test.csv"
    ex.file_out(path4, path5)

    path1 = "picture/decision_tree.png"
    path2 = "picture/decision_variable_importance.png"
    ex.process(path1, path2)

    path3 = "picture/decision_predict_result_1.csv"
    ret = ex.re
    sult(path3)

    path1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path1)
    path2 = const.DIR_FORMAT.format(const.CURRENT_DIR, path2)
    path3 = const.DIR_FORMAT.format(const.CURRENT_DIR, path3)

    str = "decision&{}#{}${}#{}".format(path1, path2, ret, path3)

    return str


if __name__ == '__main__':
    print("main")

    # path1 = "XGBoost_PDW1.csv"
    path1 = "decision_PDW.csv"
    deal(path1)
