# 1.聚类结果图,显示在实验过程中
# 名称：kmeans_cluster.png
# 路径：path_1 = "picture/kmeans_cluster.png"

# 2. 显示准确率
# 3. 对比的csv文件不需要显示
# 名称：kmeans_predict_result.csv
# 路径：path2=picture/kmeans_predict_result.csv"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
# 3.准确率，显示在实验结果中
# Accuracy 0.01902881249439009
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题


def kMeans(path):
    print('两部雷达辐射源A和B，同时开机测试，对于测量到的两部辐射源的PDW特征运用K-Means算法进行聚类实验。雷达辐射源部分参数如下'"\n"
          '开始频率（MHz）	结束频率（MHz）	射频类型	重频类型'"\n"
          '12850	            11100	   捷变150MHz32点	固定'"\n"
          '8320	                11160	   捷变160MHz32点	抖动10%	'"\n"
          )

    # 数据预处理
    clos = ['中心频点', 'PA', 'PW', 'label']
    total_df = pd.read_csv(path, encoding='UTF-8', usecols=clos)
    print("path:", path)
    data = total_df.drop("label", axis=1)
    data = np.array(data)
    ss = StandardScaler()
    data = ss.fit_transform(data)  # 标准化

    # K-means聚类
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]

    # 三维画图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('K_Means Clustering Diagram', fontsize=17)
    ax.set_xlabel('中心频点', fontsize=14)
    ax.set_ylabel('PA', fontsize=14)
    ax.set_zlabel('PW', fontsize=14)
    ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], c='lightgreen', marker='s', label='cluster 1')
    ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='orange', marker='o', label='cluster 2')
    ax.legend()
    # plt.show()
    path_1 = "picture/kmeans_cluster.png"
    plt.savefig(path_1)
    plt.show()

    # 计算准确率
    for j in range(len(label_pred)):
        if label_pred[j] == 0:
            label_pred[j] = 1
        elif label_pred[j] == 1:
            label_pred[j] = 0
    df = total_df.loc[:, 'label']
    label_true = np.array(df).tolist()
    result = np.subtract(label_pred, label_true)
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
    # Accuracy = float(count[0]) / (count[0] + count[1])
    # ret = "\033[1;31;40m准确率:{:.2%}\033[0m".format(Accuracy)
    print(ret)
    # # 输出预测值与原有分类的对比
    # y_pred = label_pred.reshape(label_pred.shape[0], 1)
    # a = pd.read_csv('data/K_Means.csv', encoding='gbk')
    # a['pred'] = y_pred
    # a.to_csv(r"picture/kmeans_predict_result.csv", mode='a', index=False)
    # # a.to_csv(r"预测结果.csv", mode='a', index=False)
    # path2 = "picture/kmeans_predict_result.csv"
    import const
    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)
    return "KMeans&{}${}".format(path_1, ret)


if __name__ == '__main__':
    print("main")


def deal(path):
    path_1 = "data/{}".format(path)
    print("path_1:", path_1)
    # path_1 = "data/K_Means1.csv"
    data = kMeans(path_1)
    print("data:", data)
    return data
