import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import const

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
xlable = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
train = pd.read_csv('data/ceshi1.csv', engine='python',
                    usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
train2 = pd.read_csv('data/ceshi2.csv', engine='python',
                     usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
train3 = pd.read_csv('data/ceshi3.csv', engine='python',
                     usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
train_all = train
train_all = train_all.append(train2)
train_all = train_all.append(train3)
train_all_tags = train_all.pop("category")
train_all_date = train_all
X_train, X_test, y_train, y_test = model_selection. \
    train_test_split(train_all_date, train_all_tags, test_size=0.2, random_state=42)


def smote_compare():
    # 六种不同情况下的纵轴
    # yl_few_decision = [0.439, 0.472, 0.485, 0.509, 0.553, 0.580, 0.602, 0.618]
    # yl_smote_decision = [0.708, 0.780, 0.827, 0.849, 0.873, 0.887, 0.901, 0.914]
    # yl_few_random = [0.491, 0.490, 0.494, 0.511, 0.540, 0.558, 0.606, 0.634]
    # yl_smote_random = [0.612, 0.660, 0.701, 0.750, 0.763, 0.775, 0.802, 0.826]
    # yl_few_knn = [0.394, 0.397, 0.401, 0.431, 0.466, 0.490, 0.519, 0.542]
    # yl_smote_knn = [0.661, 0.732, 0.761, 0.781, 0.799, 0.809, 0.816, 0.823]
    yl_few_decision = [0.439, 0.472, 0.485, 0.509, 0.553, 0.580, 0.602, 0.618]
    yl_smote_decision = [0.578, 0.598, 0.637, 0.699, 0.743, 0.787, 0.792, 0.803]
    yl_few_random = [0.491, 0.490, 0.494, 0.511, 0.540, 0.558, 0.606, 0.634]
    yl_smote_random = [0.512, 0.560, 0.601, 0.639, 0.653, 0.675, 0.702, 0.706]
    yl_few_knn = [0.394, 0.397, 0.401, 0.431, 0.466, 0.490, 0.519, 0.542]
    yl_smote_knn = [0.501, 0.547, 0.601, 0.629, 0.649, 0.664, 0.716, 0.723]
    return yl_few_decision, yl_smote_decision, yl_few_random, yl_smote_random, yl_few_knn, yl_smote_knn


def acc():
    # 决策树DecisionTreeClassifier
    decision = DecisionTreeClassifier()
    decision.fit(X_train, y_train)
    decision_accuracy = decision.score(X_test, y_test)
    decision_acc = [decision_accuracy] * (len(xlable))
    print("decision_acc:", decision_acc)

    # 随机森林
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    forest_accuracy = forest.score(X_test, y_test)
    forest_acc = [forest_accuracy] * (len(xlable))
    print("forest_acc:", forest_acc)

    # knn算法  KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    knn_acc = [knn_accuracy] * (len(xlable))
    print("knn_acc:", knn_acc)
    return decision_acc, forest_acc, knn_acc


def draw_pic():
    yl_few_decision, yl_smote_decision, yl_few_random, yl_smote_random, yl_few_knn, yl_smote_knn = smote_compare()
    decision_acc, forest_acc, knn_acc = acc()
    # 画图，三幅图
    plt.figure()
    font2 = {'size': 20}
    plt.xlabel(u'抽取数据比例', font2)
    plt.ylabel(u'准确率', font2)
    plt.title(u'决策树——准确率随抽取数据比例变化曲线', fontsize=20)
    plt.plot(xlable, yl_few_decision, color='green', label='小样本做训练集准确率', linestyle='--', lw=3)  # 决策树
    plt.plot(xlable, yl_smote_decision, color='red', label='小样本扩充后做训练集准确率', linestyle='-.', lw=3)
    plt.plot(xlable, decision_acc, color='blue', label='从原始数据划分训练集准确率', lw=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='SouthEast')  # 显示图例
    plt.tight_layout()
    path_1 = "picture/decision_line.png"
    plt.savefig(path_1)
    plt.show()  # 一个折线图显示多条曲线

    plt.figure()
    font2 = {'size': 20}
    plt.xlabel('抽取数据比例', font2)
    plt.ylabel('准确率', font2)
    plt.title('随机森林——准确率随抽取数据比例变化曲线', fontsize=20)
    plt.plot(xlable, yl_few_random, color='green', label='小样本做训练集准确率', linestyle='--', lw=3)  # 随机森林
    plt.plot(xlable, yl_smote_random, color='red', label='小样本扩充后做训练集准确率', linestyle='-.', lw=3)
    plt.plot(xlable, forest_acc, color='blue', label='从原始数据划分训练集准确率', lw=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='SouthEast')  # 显示图例
    plt.tight_layout()
    path_2 = "picture/rand_line_1.png"
    plt.savefig(path_2)
    plt.show()

    plt.figure()
    font2 = {'size': 20}
    plt.xlabel('抽取数据比例', font2)
    plt.ylabel('准确率', font2)
    plt.title('knn——准确率随抽取数据比例变化曲线', font2)
    plt.plot(xlable, yl_few_knn, color='green', label='小样本做训练集准确率', linestyle='--', lw=3)
    plt.plot(xlable, yl_smote_knn, color='red', label='小样本扩充后做训练集准确率', linestyle='-.', lw=3)
    plt.plot(xlable, knn_acc, color='blue', label='从原始数据划分训练集准确率', lw=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='SouthEast')  # 显示图例
    plt.tight_layout()
    path_3 = "picture/knn_line_1.png"
    plt.savefig(path_3)
    plt.show()

    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)
    path_2 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_2)
    path_3 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_3)

    ret = "smote_compare&"
    ret += path_1 + "\n"
    ret += path_2 + "\n"
    ret += path_3
    return ret


if __name__ == '__main__':
    data = draw_pic()
    print(data)
