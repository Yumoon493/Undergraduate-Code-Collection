#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define M 4
#define N 4
#define MaxSize 100

typedef struct {
    int i;
    int j;
    int di;
} Box;

typedef struct {
    Box data[MaxSize];
    int top;
} StType;

void InitStack(StType* st) {
    st->top = -1;
}

bool StackEmpty(StType* st) {
    return st->top == -1;
}

void Push(StType* st, Box e) {
    st->top++;
    st->data[st->top] = e;
}

void Pop(StType* st, Box* e) {
    *e = st->data[st->top];
    st->top--;
}

void GetTop(StType* st, Box* e) {
    if (!StackEmpty(st))
        *e = st->data[st->top];
}

void DestroyStack(StType* st) {
    st->top = -1;
}

void findPaths(int i, int j, int xe, int ye, int mg[M + 2][N + 2], StType* st, 
    int* shortest_length, Box path[], int length, Box shortest_path[]) {
    // 如果当前位置超出迷宫边界或是墙壁，直接返回
    if (i < 1 || i > M || j < 1 || j > N || mg[i][j] == 1) {
        return;
    }

    // 如果当前位置是出口
    if (i == xe && j == ye) {
        // 检查是否为最短路径
        if (*shortest_length == -1 || length < *shortest_length) {
            *shortest_length = length;
            memcpy(shortest_path, path, length * sizeof(Box)); // 复制最短路径

            // 添加终点到最短路径
            Box end_point;
            end_point.i = xe;
            end_point.j = ye;
            shortest_path[length] = end_point;
        }

        // 打印当前路径
        printf("路径：");
        for (int m = 0; m < length; m++) {
            printf("(%d,%d) ", path[m].i, path[m].j);
        }
        printf("(%d,%d)\n", xe, ye);
        return;
    }

    // 将当前位置标记为已访问
    mg[i][j] = 1;

    // 将当前位置加入路径
    Box current;
    current.i = i;
    current.j = j;
    current.di = -1; // 任意值，因为起点的方向不重要
    path[length] = current;

    // 尝试所有可能的方向
    for (int d = 0; d < 4; d++) {
        int di = (d + 1) % 4;
        int i1 = i, j1 = j;
        switch (di) {
        case 0: i1 = i - 1; break; // 上
        case 1: j1 = j + 1; break; // 右
        case 2: i1 = i + 1; break; // 下
        case 3: j1 = j - 1; break; // 左
        }

        findPaths(i1, j1, xe, ye, mg, st, shortest_length, path, length + 1, shortest_path); 
        // 递归探索下一个位置
    }

    // 恢复当前位置的可通行状态
    mg[i][j] = 0;
}

void mgpath(int xi, int yi, int xe, int ye, int mg[M + 2][N + 2], int* shortest_length) {
    StType st;
    InitStack(&st);

    Box path[MaxSize], shortest_path[MaxSize];

    // 开始探索所有可能的路径
    findPaths(xi, yi, xe, ye, mg, &st, shortest_length, path, 0, shortest_path);

    // 打印最短路径
    printf("最短路径：");
    for (int m = 0; m < *shortest_length + 1; m++) {
        printf("(%d,%d) ", shortest_path[m].i, shortest_path[m].j);
    }
    printf("\n");

    DestroyStack(&st);
}

int main() {
    int mg[M + 2][N + 2] = {
        {1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 0, 1, 0, 0, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1}
    };

    int shortest_length = -1;
    mgpath(1, 1, 4, 4, mg, &shortest_length);

    if (shortest_length == -1)
        printf("该迷宫问题没有解!\n");
    else
        printf("最短路径的长度为：%d\n", shortest_length+1);

    return 0;
}
