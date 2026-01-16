#include <stdio.h>
#include <stdbool.h>

#define M 5
#define N 4
#define MaxSize 100

// 方块类型定义
typedef struct {
    int i;      // 当前方块的行号
    int j;      // 当前方块的列号
    int di;     // di是下一可走相邻方位的方位号
} Box;

// 栈类型定义
typedef struct {
    Box data[MaxSize];
    int top;    // 栈顶指针
} StType;

// 初始化栈
void InitStack(StType* st) {
    st->top = -1;
}

// 判断栈是否为空
bool StackEmpty(StType* st) {
    return st->top == -1;
}

// 入栈操作
void Push(StType* st, Box e) {
    st->top++;
    st->data[st->top] = e;
}

// 出栈操作
void Pop(StType* st, Box* e) {
    *e = st->data[st->top];
    st->top--;
}

// 获取栈顶元素
void GetTop(StType* st, Box* e) {
    *e = st->data[st->top];
}

// 销毁栈
void DestroyStack(StType* st) {
    st->top = -1;
}

// 打印路径
void PrintPath(Box path[], int length) {
    printf("迷宫路径如下:\n");
    for (int i = 0; i < length; i++) {
        printf("\t(%d,%d)", path[i].i, path[i].j);
        if ((i + 1) % 5 == 0)
            printf("\n");
    }
    printf("\n");
}

// 栈求解迷宫路径
bool mgpath(int xi, int yi, int xe, int ye, int mg[M + 2][N + 2], int* shortest_length) {
    StType st;
    InitStack(&st);

    Box path[MaxSize], e;
    int i, j, di, i1, j1, k;
    bool find;

    e.i = xi; e.j = yi; e.di = -1;
    Push(&st, e);
    mg[xi][yi] = -1;

    while (!StackEmpty(&st)) {
        GetTop(&st, &e);
        i = e.i; j = e.j; di = e.di;

        if (i == xe && j == ye) {
            k = 0;
            while (!StackEmpty(&st)) {
                Pop(&st, &e);
                path[k++] = e;
            }
            if (*shortest_length == -1) {
                *shortest_length = k;
                PrintPath(path, k);
            }
            DestroyStack(&st);
            return true;
        }

        find = false;
        while (di < 4 && !find) {
            di++;
            switch (di) {
            case 0: i1 = i - 1; j1 = j; break;
            case 1: i1 = i; j1 = j + 1; break;
            case 2: i1 = i + 1; j1 = j; break;
            case 3: i1 = i; j1 = j - 1; break;
            }
            if (mg[i1][j1] == 0) find = true;
        }

        if (find) {
            st.data[st.top].di = di;
            e.i = i1; e.j = j1; e.di = -1;
            Push(&st, e);
            mg[i1][j1] = -1;
        }
        else {
            Pop(&st, &e);
            mg[e.i][e.j] = 0;
        }
    }

    DestroyStack(&st);
    return false;
}

int main() {
    int mg[M + 2][N + 2] = {
        {1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 0, 1, 0, 0, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 1, 0, 0, 0, 1},
        {1, 1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1}
    };

    int shortest_length = -1;

    printf("从指定入口到出口的所有迷宫路径为：\n");
    if (!mgpath(1, 1, 4, 4, mg, &shortest_length)) {
        printf("该迷宫问题没有解!\n");
    }
    else {
        printf("第一条最短路径的长度为：%d\n", shortest_length);
    }

    return 0;
}

