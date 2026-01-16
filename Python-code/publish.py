import paho.mqtt.client as mqtt
import time
import random
import json
import logging
from database import insert_data1
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("HealthMonitorPublisher")

# MQTT代理地址和端口
#broker = "192.168.124.63"
#broker= "172.24.7.239"#修改为当前网络的ip地址
broker = "192.168.124.71"
port = 1883
username = "admin"  # 如果代理需要认证
password = "123456"  # 如果代理需要认证

# 主题
topics = {              #使用字典来存储
    "heart_rate": "health/monitor/heart_rate",
    "blood_pressure": "health/monitor/blood_pressure",
    "temperature": "health/monitor/temperature",
    "oxygen_saturation": "health/monitor/oxygen_saturation"
}

# 创建MQTT客户端实例
client = mqtt.Client()

# 设置用户名和密码（如果需要）
client.username_pw_set(username, password)

# 连接回调
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected successfully")
    else:
        logger.error(f"Connect failed with code {rc}")

# 断开连接回调
def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning("Unexpected disconnection.")
    logger.info("Disconnected successfully")

# 发布消息回调
def on_publish(client, userdata, mid):
    logger.info(f"Message {mid} published successfully")

# 绑定回调函数
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish

# 连接到MQTT代理
client.connect(broker, port, 60)

# 启动网络循环
client.loop_start()

try:
    while True:
        # 生成随机数据
        heart_rate = random.randint(60, 100)
        #心脏收缩压
        systolic_bp = random.randint(110, 140)
        #心脏舒张压
        diastolic_bp = random.randint(70, 90)
        temperature = round(random.uniform(36.0, 37.5), 1)
        oxygen_saturation = (random.randint(90, 100))
        name=username
        #将发布的数据传入数据库中
        insert_data1(name,heart_rate,systolic_bp,diastolic_bp,temperature,oxygen_saturation)
        # 创建数据字典
        data = {
            "heart_rate": {"value": heart_rate, "unit": "bpm", "timestamp": time.time()},
            "blood_pressure": {
                "systolic": systolic_bp,
                "diastolic": diastolic_bp,
                "unit": "mmHg",
                "timestamp": time.time()
            },
            "temperature": {"value": temperature, "unit": "C", "timestamp": time.time()},
            "oxygen_saturation": {"value": oxygen_saturation, "unit": "%", "timestamp": time.time()}
        }

        # 发布消息
        for key, topic in topics.items():
            payload = json.dumps(data[key])
            result = client.publish(topic, payload, qos=1)
            # 检查消息是否发送成功
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"{key.capitalize()} data sent: {payload}")
            else:
                logger.error(f"Failed to send {key} data")

        # 每隔60秒发送一次数据
        time.sleep(10)
except KeyboardInterrupt:
    logger.info("Exiting...")

# 停止网络循环并断开连接
client.loop_stop()
client.disconnect()
