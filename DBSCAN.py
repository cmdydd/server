# -*- coding: UTF-8 -*-
# 1.聚类动态图（.gif），显示在实验过程中
# 文件名称：DBSCAN_cluster.gif
# 路径：path_1 = "picture/DBSCAN_cluster.gif"

# 2.准确率，显示在实验结果中
# 显示最后的准确率


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import os
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import const

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def dbSCAN(file_path):
    # file_path = "data/{}".format(file_path)
    print('为了验证在复杂场景下，智能算法的在线学习能力。针对三种不同类型的雷达辐射源，采用DBSCAN算法进行聚类实验，辐射源参数如下：'"\n"
          '序号	射频类型	重频类型	个数'"\n"
          '1	脉压30M	      固定	     99'"\n"
          '2	固定	     3参差1.1	 84'"\n"
          '3	固定	     抖动10%     102'"\n"
          )
    ################################数据处理###########################3
    clos = ['Actual frequency', 'PW', 'PRI', 'label']
    print(file_path)
    # total_df = pd.read_csv(file_path, encoding='UTF-8', usecols=clos)
    total_df = pd.read_csv("/home/ubuntu/PycharmProjects/radar_service/data/DBSCAN2.csv", encoding='UTF-8', usecols=clos)
    print("data")
    data = total_df.drop("label", axis=1)
    data = np.array(data)
    ss = StandardScaler()
    data = ss.fit_transform(data)  # 标准化
    # DBSCAN聚类

    result_score = 0
    for i in range(1, 285, 15):
        print(i + 1)
        pdw = data[0:i + 1, :]
        estimator = DBSCAN(eps=1.5, min_samples=10, metric='euclidean')  # 构造聚类器
        res = estimator.fit_predict(pdw)  # 聚类拟合
        lable_pred = estimator.labels_  # 获取聚类标签
        clusters = lable_pred.tolist()

        #  三维画图，输出图像
        fig = plt.figure()
        ax = Axes3D(fig)
        # 坐标轴
        ax.set_title('DBSCAN Dynamic Clustering Diagram', fontsize=17)
        ax.set_xlabel('Actual frequency', fontsize=14)
        ax.set_ylabel('PW', fontsize=14)
        ax.set_zlabel('PRI', fontsize=14)
        ax.legend(loc='NorthWestOutside')
        #  将数据点分成三部分画，在颜色上有区分度
        ax.scatter(pdw[res == -1, 0], pdw[res == -1, 1], pdw[res == -1, 2], c='red', marker='x', label='noise')
        ax.scatter(pdw[res == 0, 0], pdw[res == 0, 1], pdw[res == 0, 2], c='lightgreen', marker='s', label='cluster 1')
        ax.scatter(pdw[res == 1, 0], pdw[res == 1, 1], pdw[res == 1, 2], c='orange', marker='o', label='cluster 2')
        ax.scatter(pdw[res == 2, 0], pdw[res == 2, 1], pdw[res == 2, 2], c='blue', marker='v', label='cluster 3')
        ax.legend(loc='NorthWestOutside')
        plt.savefig('DBSCAN_Clusters_{0}.jpg'.format(i + 1), format='jpg', dpi=300)
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
        # result_score = "\033[1;31;40m准确率:{:.2%}\033[0m".format(Accuracy)
    print(result_score)

    # ##################################生成gif动图#########################################################
    import imageio,os
    # path = os.chdir(or_path)
    images = []
    filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.jpg')))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('gif.gif', images, duration=1)
    print("OK")






    path_1 = "gif.gif"

    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)

    return "DBSCAN&{}${}".format(path_1, result_score)


def deal(path):
    print("path:", path)
    file_path = "data/{}".format(path)

    data = dbSCAN(file_path)
    print(data)
    return data


if __name__ == '__main__':
    print("main")
    deal("DBSCAN1.csv")
