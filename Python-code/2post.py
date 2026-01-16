import requests

# 1. 定义POST目标URL和要提交的数据
url = "https://httpbin.org/post"
data = {
    "username": "test_user",
    "password": "test_pass123",
    "action": "login"
}

# 2. 发送POST请求（表单形式）
response = requests.post(url, data=data)

# 3. 查看状态码和响应内容
print("状态码:", response.status_code)  # POST成功通常返回200
if response.status_code == 200:
    # 解析JSON格式的响应
    result = response.json()
    print("服务器返回的数据:", result)

    # 提取特定字段（例如提交的form数据）
    print("服务器接收到的表单数据:", result["form"])
else:
    print("请求失败，状态码:", response.status_code)