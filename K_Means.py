import matplotlib.pyplot as plt
import numpy as np
import const
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题


class Example:
    def __init__(self, path):
        self.total_df = None
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        print('两部雷达辐射源A和B，同时开机测试，对于测量到的两部辐射源的PDW特征运用K-Means算法进行聚类实验。雷达辐射源部分参数如下'"\n"
              '开始频率（MHz）	结束频率（MHz）	射频类型	重频类型'"\n"
              '12850	            11100	   捷变150MHz32点	固定'"\n"
              '8320	                11160	   捷变160MHz32点	抖动10%	'"\n"
              )
        # 数据预处理
        cols = ['中心频点', 'PA', 'PW', 'label']
        total_df = pd.read_csv(path, encoding='UTF-8')
        own_feature = total_df.columns.values  # 数据集具备的特征，包含标签label

        self.feature_cols = self.get_feature(cols, own_feature)  # cols：需要的特征，返回值feature_cols: 特征交集
        self.total_df = total_df.loc[:, self.feature_cols]

        data = self.total_df.drop("label", axis=1)
        data = np.array(data)
        ss = StandardScaler()
        self.data = ss.fit_transform(data)  # 标准化
        self.estimator = KMeans(n_clusters=2)  # 构造聚类器

    def process(self, path_1):
        # K-means聚类
        # self.estimator.fit(self.data)  # 聚类
        if len(self.feature_cols) <= 1:
            return const.err_format("特征不足，无法聚类")
        self.estimator.fit(self.data)  # 聚类

        self.label_pred = self.estimator.labels_  # 获取聚类标签
        self.feature_cols.remove('label')  # 剔除标签，得到特征列表
        x0 = self.data[self.label_pred == 0]
        x1 = self.data[self.label_pred == 1]
        print("len(self.feature_cols):", len(self.feature_cols))
        if len(self.feature_cols) == 2:
            self.two_wei(x0, x1, self.feature_cols, path_1)
        elif len(self.feature_cols) == 3:
            self.three_wei(x0, x1, self.feature_cols, path_1)
        else:
            print("error:特征不足，无法聚类！")

    def two_wei(self, x0, x1, feature_cols, path_1):
        plt.figure()
        plt.title('K_Means Clustering Diagram', fontsize=17)
        plt.xlabel(feature_cols[0], fontsize=14)
        plt.ylabel(feature_cols[1], fontsize=14)
        plt.scatter(x0[:, 0], x0[:, 1], c='lightgreen', marker='s', label='cluster 1')
        plt.scatter(x1[:, 0], x1[:, 1], c='orange', marker='s', label='cluster 2')
        plt.legend()
        plt.savefig(path_1)
        plt.show()

    def three_wei(self, x0, x1, feature_cols, path_1):
        print("path_1:", path_1)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('K_Means Clustering Diagram', fontsize=17)
        ax.set_xlabel(feature_cols[0], fontsize=14)
        ax.set_ylabel(feature_cols[1], fontsize=14)
        ax.set_zlabel(feature_cols[2], fontsize=14)
        ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], c='lightgreen', marker='s', label='cluster 1')
        ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='orange', marker='o', label='cluster 2')
        ax.legend()
        plt.savefig(path_1)
        plt.show()

    def get_feature(self, feature1, feature2):
        # 两种求列表并集的方法
        feature3 = (set(feature1) - set(feature2))  # 需要但是不具备的特征
        feature4 = list(set(feature1) - feature3)  # 需要且具备的特征
        if len(feature4) <= 1:
            return "err"
        else:
            return feature4

    def acc(self):
        for j in range(len(self.label_pred)):
            if self.label_pred[j] == 0:
                self.label_pred[j] = 1
            elif self.label_pred[j] == 1:
                self.label_pred[j] = 0
        df = self.total_df.loc[:, 'label']
        self.label_true = np.array(df).tolist()
        result = np.subtract(self.label_pred, self.label_true)
        count = [0, 0]
        for k in result:
            if k == 0:
                count[0] = count[0] + 1
            else:
                count[1] = count[1] + 1
        rate = float(count[0]) / (count[0] + count[1])
        if rate < 0.5:
            rate = 1 - rate
        rate = '%.4f%%' % (rate * 100)
        ret = '准确率:{}'.format(rate)
        return ret


def deal(path):
    print("path:", path)
    path = "data/{}".format(path)
    ex = Example(path)

    path_1 = "picture/kmeans_cluster.png"
    ex.process(path_1)
    print("path_1:", path_1)
    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)

    ret = ex.acc()
    return "KMeans&{}${}".format(path_1, ret)


if __name__ == '__main__':
    print("main")
    path1 = "K_Means2.csv"
    deal(path1)
