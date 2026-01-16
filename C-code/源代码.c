//202229013023N_贾喻婷_杨辉三角_源代码
#include <stdio.h>
int main() {
    int r;
    int result; // 定义scanf函数的返回值变量
    
    // 使用do-while循环来处理输入
    do {
        printf("Enter the number of rows: "); // 提示用户输入行数
        result = scanf("%d", &r); // 读取用户输入并存储到r变量中，将scanf的返回值存储到result变量中
        
        // 清空输入缓冲区，防止非法输入影响下一次读取
        while (getchar() != '\n');
        
        // 如果scanf的返回值为0，说明输入的不是一个整数
        if (result == 0) {
            printf("Invalid input. Please enter a number.\n"); // 打印错误信息
        }
    } while (result == 0); // 如果输入的不是一个整数，继续循环
    
    
    int t[r][r]; // 创建一个二维数组用于存储杨辉三角的数字
    int i, j, count;
    
    // 生成杨辉三角
    for (i = 0; i < r; i++) {
        t[i][0] = 1; // 每行的第一个数字为1
        t[i][i] = 1; // 每行的最后一个数字为1
        
        for (j = 1; j < i; j++) {
            // 根据杨辉三角的性质计算其他位置的数字
            t[i][j] = t[i-1][j-1] + t[i-1][j];
        }
    }
    
    // 打印杨辉三角形，for嵌套
    for (i = 0; i < r; i++) {
    	// 输出每一行前面对齐用的空格
    	for (count = 0; count < 4*(r-i-1); count++) { //其实就是 r-(i+1)，r从1开始，i从0（实际代表1）开始
        	printf(" ");
    	}
    	
        for (j = 0; j <= i; j++) {
            // 输出杨辉三角形的数字
            printf("   %d   ", t[i][j]);//前后空3个，与下一行一起，数字之间空4个（前后各两个，好看），对应上面4倍
            if(i != 0) printf(" "); // 如果不是第一行，每个数字后面再输出一个空格
        }
        printf("\n");//换行，开始进入i++行循环
    }
    
    return 0;
}
