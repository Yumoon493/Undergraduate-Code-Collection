import requests
from bs4 import BeautifulSoup

# 1. 定义目标URL和请求头（伪装浏览器）
url = "https://movie.douban.com/top250"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# 2. 发送GET请求
response = requests.get(url, headers=headers)

# 3. 查看状态码和内容
print("状态码:", response.status_code)  # 成功应返回200
if response.status_code == 200:
    # 查看网页HTML内容（前500字符）
    print("响应内容（片段）:", response.text[:500])

    # 4. 解析内容提取电影标题
    soup = BeautifulSoup(response.text, "html.parser")
    titles = soup.find_all("span", class_="title")
    for title in titles:
        print(title.get_text().strip())
else:
    print("请求失败，状态码:", response.status_code)

"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>豆瓣电影 Top 250</title>
    <link rel="stylesheet" href="https://img3.doubanio.com/f/movie/7fb095135ebc3b2e/css/movie.css">
</head>
<body>
    <div id="wrapper">
        <div id="content">
            <h1>豆瓣电影 Top 250</h1>
            <div class="grid-view">
                <div class="item">
                    <div class="pic">
                        <a href="https://movie.douban.com/subject/1292052/">
                            <img src="https://img3.doubanio.com/view/photo/s_ratio_poster/public/p480747492.jpg" alt="肖申克的救赎">
                        </a>
                    </div>
                    <div class="info">
                        <div class="hd">
                            <a href="https://movie.douban.com/subject/1292052/" class="">
                                <span class="title">肖申克的救赎</span>
                                <span class="title">&nbsp;/&nbsp;The Shawshank Redemption</span>
                                <span class="other">&nbsp;/&nbsp;月黑高飞(港)  /  刺激1995(台)</span>
                            </a>
                        </div>
                        <div class="bd">
                            <p class="">
                                导演: 弗兰克·德拉邦特&nbsp;&nbsp;&nbsp;主演: 蒂姆·罗宾斯 / 摩根·弗里曼...<br>
                                1994&nbsp;/&nbsp;美国&nbsp;/&nbsp;犯罪 剧情
                            </p>
                            <div class="star">
                                <span class="rating45-t"></span>
                                <span class="rating_num" property="v:average">9.7</span>
                                <span property="v:best" content="10.0"></span>
                                <span>1090890人评价</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""