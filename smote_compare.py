import matplotlib.pyplot as plt
from pylab import mpl


def smote_compare():
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    xlable = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # 六种不同情况下的纵轴
    yl_few_decision = [0.4390217391304348, 0.47171195652173914, 0.48492753623188406, 0.509302536231884,
                       0.5537952898550724,
                       0.5795018115942029, 0.6019112318840579, 0.6178442028985507, 0.6478079710144927,
                       0.6830072463768115]
    yl_smote_decision = [0.7082608695652174, 0.7792844202898551, 0.8271195652173913, 0.848677536231884,
                         0.8728713768115942,
                         0.8867572463768116, 0.9012228260869565, 0.9142844202898551, 0.9224547101449275,
                         0.9358695652173913, ]
    yl_few_random = [0.4915126811594203, 0.4890398550724638, 0.4938768115942029, 0.5109963768115942, 0.5394474637681159,
                     0.5573550724637681, 0.6057789855072464, 0.6340126811594203, 0.6517934782608695, 0.6821648550724637]
    yl_smote_random = [0.6182065217391305, 0.6592481884057971, 0.7014764492753623, 0.7501992753623189,
                       0.7633514492753624,
                       0.7753442028985508, 0.8019384057971014, 0.826231884057971, 0.844375, 0.8636684782608696, ]
    yl_few_knn = [0.39386775362318843, 0.39676630434782606, 0.40110507246376814, 0.4308423913043478, 0.4660597826086956,
                  0.4900090579710145, 0.5180163043478261, 0.5428079710144927, 0.5684601449275363, 0.5923731884057971]
    yl_smote_knn = [0.6612590579710145, 0.7325815217391304, 0.7613768115942029, 0.7812952898550725, 0.798695652173913,
                    0.808731884057971, 0.8157427536231884, 0.8235144927536232, 0.8310960144927536, 0.8372373188405797, ]
    print("yl_few_decision：", yl_few_decision)
    print("yl_smote_decision：", yl_smote_decision)
    print("yl_few_random：", yl_few_random)
    print("yl_smote_random：", yl_smote_random)
    print("yl_few_knn：", yl_few_knn)
    print("yl_smote_knn：", yl_smote_knn)
    # 画图，三幅图
    plt.figure()
    plt.xlabel(u'抽取数据比例')
    plt.legend(u'准确率')
    plt.title(u'决策树——准确率随抽取数据比例变化曲线')
    plt.plot(xlable, yl_few_decision, color='green', label='few decision accuracy')  # 决策树
    plt.plot(xlable, yl_smote_decision, color='red', label='smote decision accuracy')
    plt.legend()  # 显示图例
    # plt.show()  # 一个折线图显示多条曲线
    path_1 = "/home/ubuntu/PycharmProjects/webTest/picture/decesion_line.png"
    plt.savefig(path_1)

    plt.figure()
    plt.xlabel('抽取数据比例')
    plt.legend('准确率')
    plt.title('随机森林——准确率随抽取数据比例变化曲线')
    plt.plot(xlable, yl_few_random, color='green', label='few random accuracy')  # 随机森林
    plt.plot(xlable, yl_smote_random, color='red', label='smote random accuracy')
    plt.legend()  # 显示图例
    # plt.show()  # 一个折线图显示多条曲线
    path_2 = "/home/ubuntu/PycharmProjects/webTest/picture/rand_line.png"
    plt.savefig(path_2)

    plt.figure()
    plt.xlabel('抽取数据比例')
    plt.legend('准确率')
    plt.title('knn——准确率随抽取数据比例变化曲线')
    plt.plot(xlable, yl_few_knn, color='green', label='few knn accuracy')  # knn
    plt.plot(xlable, yl_smote_knn, color='red', label='smote knn accuracy')
    plt.legend()  # 显示图例


    path_3 = "/home/ubuntu/PycharmProjects/webTest/picture/knn_line.png"
    plt.savefig(path_3)
    # plt.show()  # 一个折线图显示多条曲线

    ret = "smote_compare&"
    ret += path_1+"\n"
    ret += path_2+"\n"
    ret += path_3
    return ret


if __name__ == '__main__':
    data = smote_compare()
    print(data)