#include <iostream>
#include<stdio.h>
#include<math.h>
#include<time.h>
clock_t start, stop, start2, stop2, start3, stop3, start4, stop4, start5, stop5, start6, stop6, start7, stop7, start8, stop8;/*clock_t是clock( )函数返回的变量类型*/
double duration, duration2, duration3, duration4, duration5, duration6, duration7, duration8;/*记录被测函数运行时间，以秒为单位*/
int n;
double result;
int Function(int n);//以2为底n的对数
int Function2(int n);//n开根号
int Function3(int n);//n本身
int Function4(int n);//n*以2为底n的对数
int Function5(int n);//n*n
int Function6(int n);//n*n*n
int Function7(int n);//2**n
int Function8(int n);//n!
//因为要分别对每个计算以及输出过程计时，所以分成八个函数

int main()
{   /*不在测试范围内的准备工作写在clock( )调用之前*/
    start = clock();/*开始计时*/
    Function(n);//以2为底n的对数
    stop = clock();/*停止计时*/
    duration = ((double)(stop - start)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\n以2为底n的对数运行时间为%lfs\n\n", duration);

    start2 = clock();/*开始计时*/
    Function2(n);//n开根号
    stop2 = clock();/*停止计时*/
    duration2 = ((double)(stop2 - start2)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn开根号运行时间为%lfs\n\n", duration2);

    start3 = clock();/*开始计时*/
    Function3(n);//n本身
    stop3 = clock();/*停止计时*/
    duration3 = ((double)(stop3 - start3)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn本身运行时间为%lfs\n\n", duration3);

    start4 = clock();/*开始计时*/
    Function4(n);//n*以2为底n的对数
    stop4 = clock();/*停止计时*/
    duration4 = ((double)(stop4 - start4)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn*以2为底n的对数运行时间为%lfs\n\n", duration4);

    start5 = clock();/*开始计时*/
    Function5(n);//n*n
    stop5 = clock();/*停止计时*/
    duration5 = ((double)(stop5 - start5)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn^2运行时间为%lfs\n\n", duration5);

    start6 = clock();/*开始计时*/
    Function6(n);//n*n*n
    stop6 = clock();/*停止计时*/
    duration6 = ((double)(stop6 - start6)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn^3运行时间为%lfs\n\n", duration6);

    start7 = clock();/*开始计时*/
    Function7(n);//2**n
    stop7 = clock();/*停止计时*/
    duration7 = ((double)(stop7 - start7)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\n2^n运行时间为%lfs\n\n", duration7);

    start8 = clock();/*开始计时*/
    Function8(n);//n!
    stop8 = clock();/*停止计时*/
    duration8 = ((double)(stop8 - start8)) / CLOCKS_PER_SEC;//*计算运行时间*/秒钟
    printf("\nn!运行时间为%lfs\n\n", duration8);

    return 0;
}

int Function(int n)//以2为底n的对数
{
    int i = 1;
    for (i = 1; i <= n; i++)
    {
        result = log2(i);
        printf("%lf,",result);
    }
    
    return 0;
}

int Function2(int n)//n开根号
{
    int i = 1;
    for (i = 1; i <= n; i++)
    {
        result = sqrt(i);
        printf("%lf,", result);
    }

    return 0;
}

int Function3(int n)//n本身
{
    int i = 1;
    for (i = 1; i <= n; i++)
    {
        printf("%d,", i);
    }

    return 0;
}

int Function4(int n)//n*以2为底n的对数
{
    int i = 1;
    for (i = 1; i <= n; i++)
    {
        result = i*log2(i);
        printf("%lf,", result);
    }

    return 0;
}

int Function5(int n)//n*n
{
    int i = 1;
    int sum;
    for (i = 1; i <= n; i++)
    {
        sum = i * i;
        printf("%d,", sum);
    }
    return 0;
}

int Function6(int n)//n*n*n
{
    int i = 1;
    int sum;
    for (i = 1; i <= n; i++)
    {
        sum = i * i * i;
        printf("%d,", sum);
    }
    return 0;
}

int Function7(int n)//2^n
{
    int i = 1;
    int sum;
    for (i = 1; i <= n; i++)
    {
        sum = 2 ^ i;
        printf("%d,", sum);
    }
    return 0;
}

int Function8(int n)//n!
{
    int i = 1;
    int sum = 1;
    for (i = 1; i <= n; i++)
    {
        sum = sum * i;
        printf("%d,", sum);
    }
    return 0;
}