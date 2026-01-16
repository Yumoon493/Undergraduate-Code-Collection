import pymysql

# 数据库连接配
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'fight040903',  # 替换为你的MySQL root用户的密码
    'database': 'medical_system',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
# 数据库初始化
#构建存储设备端传入数据的数据库
def init_db1():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT NOT NULL,
                heart_rate INT NOT NULL,
                systolic_bp INT NOT NULL,
                diastolic_bp INT NOT NULL,
                temperature INT NOT NULL,
                oxygen_saturation INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP     
            )
            ''')
        connection.commit()
    finally:
        connection.close()


#######创建数据库display
def init_db2():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS display (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT ,
                heart_rate TEXT ,
                blood_pressure TEXT ,
                temperature TEXT ,
                oxygen_saturation TEXT ,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP     
            )
            ''')
        connection.commit()
    finally:
        connection.close()
    #######创建数据库doctor_diagnosis


def init_db3():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor_diagnosis (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP     
            )
            ''')
        connection.commit()
    finally:
        connection.close()
    #####创建数据库doctors


def init_db4():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT NOT NULL,
                age INT NOT NULL,
                account INT NOT NULL,
                password INT NOT NULL    
            )
            ''')
        connection.commit()
    finally:
        connection.close()


#########构建数据库patients
def init_db5():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT NOT NULL,
                age INT NOT NULL,
                gender TEXT NOT NULL   
            )
            ''')
        connection.commit()
    finally:
        connection.close()


#########数据库的初始化
init_db1()
init_db2()
init_db3()
init_db4()
init_db5()


########
########对数据库进行操作的函数，可通过导入包的方式调用以下函数
#######对数据库 health_data进行操作
def insert_data1(a, b, c, d, e, f):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
                INSERT INTO health_data (name,heart_rate, systolic_bp,diastolic_bp, temperature, oxygen_saturation) 
                VALUES (%s, %s, %s, %s,%s,%s)
                ''', (a, b, c, d, e, f))
        connection.commit()
    finally:
        connection.close()


#######对数据库display进行操作
def insert_data2(columns, values):
    connection = pymysql.connect(**db_config)

    # 构建列名和占位符的字符串
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * 5)

    # 创建 SQL 语句
    sql = f'INSERT INTO display ({columns_str}) VALUES ({placeholders})'

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, values)
        connection.commit()
    finally:
        connection.close()


########对数据库doctor_diagnosis进行操作
def insert_data3(a, b):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            INSERT INTO doctor_diagnosis (name, diagnosis) 
            VALUES (%s, %s)
            ''', (a, b))
        connection.commit()
    finally:
        connection.close()
