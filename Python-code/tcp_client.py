import socket

def send_request(host="127.0.0.1", port=8080):
    # 创建套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 创建HTTP GET请求
    request = f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
    client_socket.sendall(request.encode('utf-8'))

    # 接收服务器响应
    response = client_socket.recv(4096)
    print(f"Received response:\n{response.decode('utf-8')}")

    client_socket.close()


if __name__ == "__main__":
    send_request()
