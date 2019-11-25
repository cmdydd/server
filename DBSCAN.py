import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import const

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def dbSCAN(file_path):
    print('为了验证在复杂场景下，智能算法的在线学习能力。针对三种不同类型的雷达辐射源，采用DBSCAN算法进行聚类实验，辐射源参数如下：'"\n"
          '序号	射频类型	重频类型	个数'"\n"
          '1	脉压30M	      固定	     99'"\n"
          '2	固定	     3参差1.1	 84'"\n"
          '3	固定	     抖动10%     102'"\n"
          )
    cols = ['Actual frequency', 'PW', 'PRI', 'label']
    # cols = ['Actual frequency', 'PRI', 'label']
    total_df = pd.read_csv(file_path, encoding='UTF-8')
    own_feature = total_df.columns.values  # 数据集具备的特征，包含标签label
    feature_cols = get_feature(cols, own_feature)  # cols：需要的特征，返回值feature_cols: 特征交集
    if len(feature_cols) <= 2:
        return const.err_format.format("特征不足，无法聚类！")
    total_df = total_df.loc[:, feature_cols]
    data = total_df.drop("label", axis=1)
    data = np.array(data)
    ss = StandardScaler()
    data = ss.fit_transform(data)  # 标准化

    # DBSCAN聚类
    feature_cols.remove('label')  # 聚类特征
    result_score = 0
    for i in range(1, 285, 15):
        print(i + 1)
        pdw = data[0:i + 1, :]
        estimator = DBSCAN(eps=1.5, min_samples=10, metric='euclidean')  # 构造聚类器
        res = estimator.fit_predict(pdw)  # 聚类拟合
        lable_pred = estimator.labels_  # 获取聚类标签
        clusters = lable_pred.tolist()

        if len(feature_cols) == 2:
            plt.figure()
            plt.title('DBSCAN Dynamic Clustering Diagram', fontsize=17)
            plt.xlabel(feature_cols[0], fontsize=14)
            plt.ylabel(feature_cols[1], fontsize=14)
            plt.legend(loc='NorthWestOutside')
            #  将数据点分成三部分画，在颜色上有区分度
            plt.scatter(pdw[res == -1, 0], pdw[res == -1, 1], c='red', marker='x', label='noise')
            plt.scatter(pdw[res == 0, 0], pdw[res == 0, 1], c='lightgreen', marker='s', label='cluster 1')
            plt.scatter(pdw[res == 1, 0], pdw[res == 1, 1], c='orange', marker='o', label='cluster 2')
            plt.scatter(pdw[res == 2, 0], pdw[res == 2, 1], c='blue', marker='v', label='cluster 3')
            plt.legend(loc='NorthWestOutside')
            plt.savefig('picture/DBSCAN_Clusters_{0}.jpg'.format(i + 1), format='jpg', dpi=300)
            plt.show()
        elif len(feature_cols) == 3:
            #  三维画图，输出图像
            fig = plt.figure()
            ax = Axes3D(fig)
            # 坐标轴
            ax.set_title('DBSCAN Dynamic Clustering Diagram', fontsize=17)
            ax.set_xlabel(feature_cols[0], fontsize=14)
            ax.set_ylabel(feature_cols[1], fontsize=14)
            ax.set_zlabel(feature_cols[2], fontsize=14)
            ax.legend(loc='NorthWestOutside')
            #  将数据点分成三部分画，在颜色上有区分度
            ax.scatter(pdw[res == -1, 0], pdw[res == -1, 1], pdw[res == -1, 2], c='red', marker='x', label='noise')
            ax.scatter(pdw[res == 0, 0], pdw[res == 0, 1], pdw[res == 0, 2], c='lightgreen', marker='s',
                       label='cluster 1')
            ax.scatter(pdw[res == 1, 0], pdw[res == 1, 1], pdw[res == 1, 2], c='orange', marker='o', label='cluster 2')
            ax.scatter(pdw[res == 2, 0], pdw[res == 2, 1], pdw[res == 2, 2], c='blue', marker='v', label='cluster 3')
            ax.legend(loc='NorthWestOutside')
            plt.savefig('picture/DBSCAN_Clusters_{0}.jpg'.format(i + 1), format='jpg', dpi=300)
            plt.show()
        # 计算准确率
        for j in range(len(lable_pred)):
            if lable_pred[j] == 0:
                lable_pred[j] = 0
            elif lable_pred[j] == 1:
                lable_pred[j] = 1
            elif lable_pred[j] == 2:
                lable_pred[j] = 2
        df = total_df.loc[0:i, 'label']
        label_true = np.array(df).tolist()
        result = np.subtract(lable_pred, label_true)
        count = [0, 0]
        for k in result:
            if k == 0:
                count[0] = count[0] + 1
            else:
                count[1] = count[1] + 1
        rett = float(count[0]) / (count[0] + count[1])
        rett = '%.4f%%' % (rett * 100)
        result_score = '准确率:{}'.format(rett)
    print(result_score)

    # 生成gif动图
    import imageio, os
    images = []
    filenames = sorted((fn for fn in os.listdir('picture/.') if fn.endswith('.jpg')))
    for filename in filenames:
        filename = "{}/{}".format("picture", filename)
        images.append(imageio.imread(filename))
    path_1 = "picture/DBSCAN.gif"
    imageio.mimsave(path_1, images, duration=1)
    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)

    os.system("rm -rf picture/DBSCAN_Clusters_*")

    return "DBSCAN&{}${}".format(path_1, result_score)


def get_feature(feature1, feature2):
    # 两种求列表并集的方法
    feature3 = (set(feature1) - set(feature2))  # 需要但是不具备的特征
    feature4 = list(set(feature1) - feature3)  # 需要且具备的特征
    if len(feature4) <= 1:
        return "err"
    else:
        return feature4


def deal(path):
    print("path:", path)
    file_path = "data/{}".format(path)

    data = dbSCAN(file_path)
    print(data)
    return data


if __name__ == '__main__':
    print("main")
    deal("DBSCAN1.csv")
