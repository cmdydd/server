from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn import model_selection
import logging


class TrainSmote:
    def __init__(self, i):
        """

        :param i: 抽取数据的比例  测试集 / 总数
        """

        self.first_file = "data/ceshi1.csv"
        self.second_file = "data/ceshi2.csv"
        self.third_file = "data/ceshi3.csv"

        self.train_fi, self.train_se, self.train_th, self.train_all, self.train_all_tags = self.load_file()

        self.select_fi, self.select_se, self.select_th = self.select_train(i)

        self.fill_data, self.fill_tags = self.fill_train()

    def load_file(self):
        train_fi = pd.read_csv(self.first_file, engine='python',
                               usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
        train_se = pd.read_csv(self.second_file, engine='python',
                               usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
        train_th = pd.read_csv(self.third_file, engine='python',
                               usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])

        # 合并所有数据
        train_all = train_fi
        train_all = train_all.append(train_se)
        train_all = train_all.append(train_th)

        train_all_tags = train_all.pop("category")
        return train_fi, train_se, train_th, train_all, train_all_tags

    def select_train(self, i):
        _, front = model_selection.train_test_split(self.train_fi, test_size=i)
        _, now = model_selection.train_test_split(self.train_se, test_size=i)
        _, after = model_selection.train_test_split(self.train_th, test_size=i)
        print("小样本抽取比例:", i)
        return front, now, after

    def fill_train(self):
        """
        扩充不平衡样本
            规则： 训练集1 + 训练集2（部分） 扩充成 fill_fi
        :return:
        """
        fill_train_se, fill_tag_se = self.fill_train_simple(self.train_fi, self.select_se)
        fill_train_th, fill_tag_th = self.fill_train_simple(self.train_se, self.select_th)
        fill_train_fi, fill_tag_fi = self.fill_train_simple(self.train_th, self.select_fi)

        fill_data = self.append_array(fill_train_fi, fill_train_se, fill_train_th, 2)
        # print("len(fill_data):", len(fill_data))
        fill_tag = self.append_array(fill_tag_fi, fill_tag_se, fill_tag_th, 2)
        # print("len(fill_tag):", len(fill_tag))

        return fill_data, fill_tag

    @staticmethod
    def fill_train_simple(train_data, select_data):
        smote = SMOTE()  # 扩充算法

        new_train = train_data.append(select_data)
        new_tags = new_train.pop('category')
        fill_data, fill_tags = smote.fit_sample(new_train, new_tags)

        return fill_data, fill_tags

    @staticmethod
    def append_array(data_fi, data_se, data_th, num):
        """

        :param data_fi:  扩展集1
        :param data_se:  扩展集2
        :param data_th:  扩展集3
        :param num:
        :return:
        """
        count = int(len(data_fi) / num)
        ret = data_fi[count:]
        count = int(len(data_se) / num)
        ret = np.concatenate((ret, data_se[count:]))
        count = int(len(data_th) / num)
        ret = np.concatenate((ret, data_th[count:]))

        return ret

    def test_train(self):
        """
            获取测试结果
            准确度
        :return: accuracy :准确度
        """

        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(self.fill_data, self.fill_tags)
        forest_accuracy = forest.score(self.train_all, self.train_all_tags)
        print("抽取比例:", count, ":", forest_accuracy)
        return forest_accuracy


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    logging.info("start train !!")
    count = 0.05
    while count < 0.6:
        train = TrainSmote(count)
        accuracy = train.test_train()
        # logging.info("train smote success accuracy :%.10f" % accuracy)
        count = count + 0.05
