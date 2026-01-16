import paho.mqtt.client as mqtt
import json
import logging
from database import insert_data2
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("HealthMonitorSubscriber")

# MQTT代理地址和端口
broker = "192.168.124.71"
port =1883
username = "root"  # 如果代理需要认证
password = "123456"  # 如果代理需要认证

# 主题
topics = ["health/monitor/heart_rate",
          "health/monitor/blood_pressure",
          "health/monitor/temperature",
          "health/monitor/oxygen_saturation"]

# 连接回调
def on_connect(client, userdata, flags, rc): #rc=0 连接成功
    if rc == 0:
        logger.info("Connected successfully")
        # 连接成功后订阅主题
        for topic in topics:
            client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    else:
        logger.error(f"Connect failed with code {rc}")

# 断开连接回调
def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning("Unexpected disconnection.")
    logger.info("Disconnected successfully")

# 消息回调
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode()) #msg，二进制格式，处理，赋值payload
        logger.info(f"Received message on topic {msg.topic}: {json.dumps(payload, indent=2)}")
        colums=["name",'heart_rate','blood_pressure','temperature','oxygen_saturation']
        values=[None,None,None,None,None]
        values[0]=username
        # 根据主题处理数据
        if msg.topic == "health/monitor/heart_rate":
            string1 =handle_heart_rate(payload)
            values[1]=string1
        elif msg.topic == "health/monitor/blood_pressure":
            string2=handle_blood_pressure(payload)
            values[2]=string2
        elif msg.topic == "health/monitor/temperature":
            string3=handle_temperature(payload)
            values[3]=string3
        elif msg.topic == "health/monitor/oxygen_saturation":
            string4=handle_oxygen_saturation(payload)
            values[4]=string4
        insert_data2(colums,values)

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON message from topic {msg.topic}")

# 数据处理函数
def handle_heart_rate(data):
    heart_rate = data["value"]
    if heart_rate < 60 or heart_rate > 100:
        string2=f"Abnormal heart rate detected: {heart_rate} bpm"
        logger.warning(f"Abnormal heart rate detected: {heart_rate} bpm")
        return string2
    else:
        logger.info(f"Normal heart rate: {heart_rate} bpm")
        string2=f"Normal heart rate: {heart_rate} bpm"
        return string2

def handle_blood_pressure(data):
    systolic = data["systolic"]
    diastolic = data["diastolic"]
    if systolic > 140 or diastolic > 90:
        string2=f"High blood pressure detected: {systolic}/{diastolic} mmHg"
        logger.warning(f"High blood pressure detected: {systolic}/{diastolic} mmHg")
        return string2
    elif systolic < 90 or diastolic < 60:
        string2=f"Low blood pressure detected: {systolic}/{diastolic} mmHg"
        logger.warning(f"Low blood pressure detected: {systolic}/{diastolic} mmHg")
        return string2
    else:
        string2=f"Normal blood pressure: {systolic}/{diastolic} mmHg"
        logger.info(f"Normal blood pressure: {systolic}/{diastolic} mmHg")
        return string2

def handle_temperature(data):
    temperature = data["value"]
    if temperature < 36.0 or temperature > 37.5:
        string2=f"Abnormal temperature detected: {temperature} C"
        logger.warning(f"Abnormal temperature detected: {temperature} C")
        return string2
    else:
        string2=f"Normal temperature: {temperature} C"
        logger.info(f"Normal temperature: {temperature} C")
        return string2

def handle_oxygen_saturation(data):
    oxygen_saturation = data["value"]
    if oxygen_saturation < 90:
        string2=f"Abnormal oxygen saturation: {oxygen_saturation}%"
        logger.warning(f"Abnormal oxygen saturation: {oxygen_saturation}%")
        return string2
    else:
        string2=f"Normal oxygen saturation: {oxygen_saturation}%"
        logger.info(f"Normal oxygen saturation: {oxygen_saturation}%")
        return string2

# 创建MQTT客户端实例
client = mqtt.Client()
# 设置用户名和密码（如果需要）
client.username_pw_set(username, password)

# 绑定回调函数
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

# 连接到MQTT代理
client.connect(broker, port, 120)

# 启动网络循环
client.loop_forever()
