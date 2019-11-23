# 2.一个树形的png文件，需要打开,显示在实验过程中
# 名称：XGBoost_tree.png
# 路径： path1 = "picture/XGBoost_tree.png"

# 2.特征重要性图,显示在实验过程中
# 名称：XGBoost_variable_importance_2.png
# 路径： path2 = "picture/XGBoost_variable_importance.png"

# 3.分选准确率，显示在实验结果中
# Accuracy:90%


# 4.csv文件“预测结果”
# 名称：XGBoost_predict_result.csv
# 路径：res.to_csv(r"picture/XGBoost_predict_result.csv",
#            mode='a', index=False)
# 路径：path3="picture/XGBoost_predict_result.csv"

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
from sklearn import metrics
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier

import const

warnings.filterwarnings('ignore')

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def XGBoost(file_path, path_1, path_2, path_3, col):
    # 计算特征重要性
    def plot_feature_importance(model, X_train, path2):
        feat_labels = X_train.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X_train.shape[1]):
            print("%2d)  %-*s  %f" % (f + 1, 30,
                                      feat_labels[f],
                                      importances[indices[f]]))
        plt.title('特征重要性分析', fontsize=20)
        font2 = {'size': 20}
        plt.xlabel(u'特征变量', font2)
        plt.ylabel(u'重要度', font2)
        plt.xticks(range(X_train.shape[1]),
                   feat_labels, rotation=0, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([-1, X_train.shape[1]])
        plt.bar(range(X_train.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')
        plt.tight_layout()
        plt.savefig(path2)
        plt.show()

    # XGBoost结果可视化

    def ceate_feature_map(features, fmap_filename, path_1):
        outfile = open(fmap_filename, 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        from xgboost import plot_tree
        plot_tree(model, num_trees=0, fmap=fmap_filename)
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        fig.savefig(path_1)
        from PIL import Image
        im = Image.open(path_1)
        im.show()

    # 读入表格文件函数
    train = pd.read_csv(file_path, encoding='UTF-8')
    X = train
    y = X.pop('label')
    feature_cols = col
    model = XGBClassifier()
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.3)
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    # 计算评价指标
    accuracy = model.score(X_test, Y_test)
    accuracy = '%.4f%%' % (accuracy * 100)
    recall = metrics.recall_score(Y_test, y_pred, average='macro')
    recall = '%.4f%%' % (recall * 100)
    f1 = metrics.f1_score(Y_test, y_pred, average='weighted')
    f1 = '%.4f%%' % (f1 * 100)
    ret = "准确率:{}\n召回率:{}\nF-1 score:{}".format(accuracy, recall, f1)
    # ret = "\033[1;31;40m准确率:{:.2%}\033[0m\nRecall:{}\nF-1 score:{}".format(accuracy, recall, f1)

    plot_feature_importance(model, X_train, path_2)

    # 输出预测值与原有分类的对比
    y_pred = y_pred.reshape(y_pred.shape[0], 1)
    train1 = pd.read_csv(file_path, encoding='UTF-8')
    res = train1.loc[Y_test.index]
    res['pred'] = y_pred
    res = res.iloc[0:500]
    res.to_csv(path_3)

    fmap_filename = "picture/xgb_2.fmap"
    ceate_feature_map(feature_cols, fmap_filename, path_1)
    return ret


def deal(seq, file_path):
    file_path = "data/{}".format(file_path)
    path_1 = "picture/XGBoost_tree.png"
    path_2 = "picture/XGBoost_variable_importance.png"
    path_3 = "picture/XGBoost_predict_result.csv"
    if seq == "a":
        feature = ['TOA', 'PA', 'RF_START', 'DOA']  # 准确率高的
    elif seq == "b":
        feature = ['Beam', 'PA', 'PAL', 'PA_MID']  # 准确率低的
    elif seq == "c":
        feature = ['PAL', 'PAR', 'PA3']  # 准确率最低的
    else:
        print("特征传输错误")
        return "error"

    path_1 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_1)
    path_2 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_2)
    path_3 = const.DIR_FORMAT.format(const.CURRENT_DIR, path_3)
    ret = XGBoost(file_path, path_1, path_2, path_3, feature)

    str = "XGBoost&{}#{}${}#{}".format(path_1, path_2, ret, path_3)
    return str


if __name__ == '__main__':
    file_path = "XGBoost_PDW1.csv"
    # data = deal("a", file_path)
    # print("done 1")
    data_a = deal("b", file_path)
    # print("done 2")
    # data_c = deal("c", file_path)
    # print(data)
    # print(data_a)
    # print(data_c)
