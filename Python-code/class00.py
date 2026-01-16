"""
class Car:
    # 类属性
    color = "red"  # 默认颜色
    speed = 0      # 初始速度

    # 类方法（行为）
    def start(self):
        print("汽车启动")

    def brake(self):
        print("汽车刹车")
        self.speed = 0  # 停止汽车，速度设为 0
# 创建汽车对象
my_car = Car()

# 访问对象的属性
print(my_car.color)  # 输出 "red"
print(my_car.speed)  # 输出 0

# 调用对象的方法
my_car.start()  # 输出 "汽车启动"
my_car.brake()  # 输出 "汽车刹车"

"""
from math import gcd

class Fraction:
    def __init__(self,num,den):
        if den == 0:
            raise Exception("Den can not be zero.")
        common = gcd(num,den)
        self.num = num // common
        self.den = den // common
        if self.den < 0:
            self.num = -self.num
            self.den = -self.den

    def add(self,other):
        if not isinstance(other,Fraction):
            raise TypeError("Must be a fraction.")
        if not isinstance(self,Fraction):
            raise TypeError("Must be a fraction.")
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den * other.den
        return Fraction(new_num,new_den)

    def multiply(self,other):
        if not isinstance(other,Fraction):
            raise TypeError("Must be a fraction.")
        if not isinstance(self,Fraction):
            raise TypeError("Must be a fraction.")
        new_num = self.num * other.num
        new_den = self.den * other.den
        return Fraction(new_num,new_den)

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __str__(self):
        return f"{self.num}/{self.den}"

    def __eq__(self, other):
        if not isinstance(other, Fraction):
            return False
        return self.num == other.num and self.den == other.den

    def __lt__(self, other):
        if not isinstance(other, Fraction):
            raise TypeError("Must be a fraction.")
        return self.num * other.den < other.num * self.den

if __name__ == "__main__":
    f1 = Fraction(3, 9)  # 化简为 1/3
    f2 = Fraction(2, 6)  # 化简为 1/3
    f3 = Fraction(1, -4) # 化简为 -1/4

    print(f"f1: {f1}")  # 输出 1/3
    print(f"f2: {f2}")  # 输出 1/3
    print(f"f3: {f3}")  # 输出 -1/4

    print("f1 + f2 =", f1 + f2)  # 加法
    print("f1 * f3 =", f1 * f3)  # 乘法
    print("f1 == f2:", f1 == f2)  # 相等性比较
    print("f1 < f3:", f1 < f3)    # 比较大小
