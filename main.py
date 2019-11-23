import http.server
from urllib import parse

import DBSCAN
import K_Means
import XGBoost
# import XGBoost_1
import decision
import random_r
import smote_compare
import smote_expansion_data

PORT = 8080

callback = None


class AjaxHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = parse.urlparse(self.path)
        data = parsed_path.query  # 前端传输数据
        # array = data.split('=')
        arr = self.prase_data(data)
        print("arr:", arr)
        self.start(arr)

    def prase_data(self, b):
        ret = []
        array1 = b.split('&')
        for i in array1:
            array2 = i.split('=')
            if len(array2) < 2:
                print("前端数据划分出错")
            else:
                ret.append(array2[1])
        return ret

    def start(self, data):
        print("from html req:", data)
        result = ""
        if len(data) <= 1:
            result = "<p>出错</p>"
        elif data[0] == "expansion_data":
            result = smote_expansion_data.expansion_data()
        elif data[0] == "smote_compare":
            result = smote_compare.draw_pic()
        elif data[0] == "DBSCAN":
            result = DBSCAN.deal(data[2])
            # result = "DBSCAN&picture/DBSCAN_cluster.gif$Accuracy:0.9963235294117647"
        elif data[0] == "decision":
            result = decision.deal(data[2])
            # result = "decision&picture/decision_tree.png#picture/decision_variable_importance.png$Accuracy:0.9999922970859876\nRecall:0.9999908273711247\nF-1 score:0.9999922981587652#picture/decision_predict_result.csv"
        elif data[0] == "random_forest":
            result = random_r.d(data[1], data[2])
            # result = test.Example(path, feature_cols)
            # result = "randomforest&picture/random_tree.png#picture/random_variable_importance.png$Accuracy:0.9954548172956162\nRecall:0.9864977747506242\nF-1 score:0.995440673267568#picture/random_predict_result.csv"
        elif data[0] == "XGBoost":
            result = XGBoost.deal(data[1], data[2])
            # result = "XGBoost&picture/XGBoost_tree.png#picture/XGBoost_variable_importance.png$Accuracy:0.9821083268544545\nRecall:0.9427370344236331\nF-1 score:0.9818209736497626#picture/XGBoost_predict_result.csv"
        elif data[0] == "KMeans":
            print("data[2]:", data[2])
            result = K_Means.deal(data[2])
            # result = "KMeans&picture/kmeans_cluster.png$Accuracy:0.9809711875056099#picture/kmeans_predict_result.csv"
        self.send_response(200)
        self.send_header("Content-type", "text/html;charset = UTF-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        print("send to html response : {}".format(result))
        self.wfile.write(result.encode())


if __name__ == "__main__":
    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), AjaxHandler)
        print("HTTP server is starting at port " + repr(PORT) + '...')
        print("Press ^C to quit")
        server.serve_forever()
    except KeyboardInterrupt:
        print("^Shutting down server...")
        server.socket.close()
