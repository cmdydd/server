import http.server
from urllib import parse

import smote_expansion_data
import smote_compare

PORT = 8080

callback = None


class AjaxHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = parse.urlparse(self.path)
        data = parsed_path.query  # 前端传输数据
        array = data.split('=')
        self.start(array)

    def start(self, data):
        result = ""
        if data[1] == "expansion_data":
            result = smote_expansion_data.expansion_data()
        elif data[1] == "smote_compare":
            result = smote_compare.smote_compare()
        self.send_response(200)
        self.send_header("Content-type", "text/html;charset = UTF-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
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
