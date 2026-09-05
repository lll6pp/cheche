from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

class MyHTTPServer(HTTPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]

server = MyHTTPServer(("127.0.0.1", 8080), SimpleHTTPRequestHandler)
print("Frontend server running at http://127.0.0.1:8080 ...")
server.serve_forever()