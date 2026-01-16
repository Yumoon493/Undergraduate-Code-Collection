#include<stdio.h>
#include<time.h>
/*要用clock( )函数必须要包含time.h*/
clock_t start, stop, start2, stop2;/*clock_t是clock( )函数返回的变量类型*/
double duration, duration2;/*记录被测函数运行时间，以秒为单位*/
int n;
int Function(int n);
int Function2(int n);

int main()
{   /*不在测试范围内的准备工作写在clock( )调用之前*/
    start = clock();/*开始计时*/
    Function(n);//累加法
    stop = clock();/*停止计时*/
    duration = ((double)(stop - start)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    /*其他不在测试范围的处理写在后面，例如输出duration的值*/
    printf("运行时间为%lfs\n", duration);

    start2 = clock();/*开始计时*/
    Function2(n);//高斯法
    stop2 = clock();/*停止计时*/
    duration2 = ((double)(stop2 - start2)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("运行时间为%lfs\n", duration2);
    return 0;
}

int Function(int n)//实验题1:求1~n连续整数和，累加法
{
    int i = 1;
    int sum = 0;
    for (i = 1; i <= n; i++)
    {
        sum = i + sum;
    }
    printf("和为%d\n", sum);
    return sum;
}

int Function2(int n)//高斯法
{
    int sum;
    sum = n * (n + 1) / 2;
    printf("和为%d\n", sum);
    return sum;
}


