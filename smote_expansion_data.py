from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn import model_selection
from collections import Counter


def expansion_data():
    train_fi = pd.read_csv("./ceshi1.csv", engine='python',
                           usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
    train_se = pd.read_csv("./ceshi2.csv", engine='python',
                           usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
    train_th = pd.read_csv("./ceshi3.csv", engine='python',
                           usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
    # 合并所有数据
    train_all = train_fi
    train_all = train_all.append(train_se)
    train_all = train_all.append(train_th)
    train_all_tags = train_all.pop("category")
    # 小样本抽取
    i = 0.05
    data = "expansion_data&"
    while i < 0.5:
        _, select_data_one = model_selection.train_test_split(train_fi, test_size=i)
        _, select_data_two = model_selection.train_test_split(train_se, test_size=i)
        _, select_data_three = model_selection.train_test_split(train_th, test_size=i)
        # print("select_data_three:", select_data_three)

        # 扩充不平衡样本
        smote = SMOTE()  # 扩充算法
        new_train_fi = train_fi.append(select_data_two)
        new_train_se = train_se.append(select_data_three)
        new_train_th = train_th.append(select_data_one)

        new_tags_fi = new_train_fi.pop('category')
        new_tags_se = new_train_se.pop('category')
        new_tags_th = new_train_th.pop('category')

        # data += "小样本抽取比例：%d 扩充前样本数目：{ 0:%d, 1:%d } \n" % (i, Counter(new_tags_fi).items())
        # data += "小样本抽取比例：%d 扩充前样本数目：%s \n" % (i, Counter(new_tags_se).items())
        # data += "小样本抽取比例：%d 扩充前样本数目：%s \n" % (i, Counter(new_tags_th).items())


        fill_data_fi, fill_tags_fi = smote.fit_sample(new_train_fi, new_tags_fi)
        fill_data_se, fill_tags_se = smote.fit_sample(new_train_se, new_tags_se)
        fill_data_th, fill_tags_th = smote.fit_sample(new_train_th, new_tags_th)

        data += format_data(i, new_tags_fi, "扩充前样本数目")
        data += format_data(i, new_tags_se, "扩充前样本数目")
        data += format_data(i, new_tags_th, "扩充前样本数目")
        print("扩充前样本数目 doing....")

        data += format_data(i, fill_tags_fi,"不平衡样本中样本数目")
        data += format_data(i, fill_tags_se,"不平衡样本中样本数目")
        data += format_data(i, fill_tags_th,"不平衡样本中样本数目")
        print("不平衡样本中样本数目 doing....")
        # data += "小样本抽取比例：%d 不平衡样本中样本数目：%s \n" % (i, Counter(fill_tags_fi).items())
        # data += "小样本抽取比例：%d 不平衡样本中样本数目：%s \n" % (i, Counter(fill_tags_se).items())
        # data += "小样本抽取比例：%d 不平衡样本中样本数目：%s \n" % (i, Counter(fill_tags_th).items())
        i = i + 0.05
    return data


def format_data(i, data_map, pre_txt):
    ret_map = Counter(data_map)
    ret_keys = ret_map.keys()
    ret = "小样本抽取比例:%.2f %s:" % (i, pre_txt)

    for i in ret_keys:
        ret += '%d:%d ' % (i, ret_map[i])
    ret += "\n"
    return ret


if __name__ == '__main__':
    data = expansion_data()
    print(data)
