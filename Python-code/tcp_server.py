import socket

# 创建HTML响应内容
HTML_CONTENT = """
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTTP Server</title>
</head>
<body>
    <h1>Welcome to the HTTP Server</h1>
    <p>This is a sample response from the server.</p>
</body>
</html>
"""

def start_server(host="127.0.0.1", port=8080):
    # 创建套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server started at {host}:{port}")

    while True:
        # 接收客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        # 接收请求数据
        request_data = client_socket.recv(1024)
        print(f"Received request:\n{request_data.decode('utf-8')}")

        # 发送HTML响应
        client_socket.sendall(HTML_CONTENT.encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    start_server()
