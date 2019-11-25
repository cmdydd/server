import http.server
from urllib import parse

import DBSCAN
import K_Means
import XGBoost
import const
import decision
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
        print("data:", arr)
        self.start(arr)

    def prase_data(self, b):
        ret = {}
        array1 = b.split('&')
        for i in array1:
            array2 = i.split('=')
            if len(array2) < 2:
                print("前端数据划分出错")
            else:
                ret[array2[0]] = array2[1]
        return ret

    def start(self, data):
        print("from html req:", data)
        result = ""
        if len(data) <= 1:
            result = "<p>出错</p>"
        else:
            name = data.get("name", "")
            if name == "expansion_data":
                result = smote_expansion_data.expansion_data()
            elif name == "smote_compare":
                result = smote_compare.draw_pic()
            elif name == "DBSCAN":
                result = DBSCAN.deal(data.get("all_file", ""))
            elif name == const.DECISION or name == const.RANDOM_FOREST:
                result = decision.deal(data)
            elif name == "XGBoost":
                result = XGBoost.deal_real(data)
            elif name == "KMeans":
                result = K_Means.deal(data["all_file"])
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
