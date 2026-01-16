// 学生信息系统.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//学生信息管理系统：学生：学号，姓名，选修课，实验课，必修课，总分合集
//系统功能：增加，删除，修改，查找，成绩统计
#include <iostream>
#include <conio.h>
#define LEN sizeof(struct student)//结构体长度命名
#define FORMAT "%-8d%-15s%-12.1lf%-12.1lf%-12.1lf%-12.1lf\n"//???
#define DATA stu[i].num,stu[i].name,stu[i].elec,stu[i].expe,stu[i].requ,stu[i].sum//下面常用前面定义方便输出

struct student
{
    int num;//学号
    char name[15];//姓名 不超过15个字符吗，其他的无限制？
    double elec;//选修课成绩
    double expe;// 实验课成绩
    double requ;//必修课成绩
    double sum;//学生总分
};
struct student stu[50];//50个叫stu[n]的结构体变量
//系统功能，引入函数
void menu();//主菜单
void in();//学生信息录入
void search();//学生信息查找
void del();//学生信息删除
void modify();//学生信息修改

int main()
{
    int n;//用变量控制选择使用函数
    menu();
    scanf("%d", &n);
    while (n)//1.while上面一行和里面最后一张衔接，while循环
    {
        switch (n)
        {
        case 1:in(); break;
        case 2:search(); break;
        case 3:del(); break;
        case 4:modify(); break;
        default:break;
        }
        _getch();
        menu();//执行操作后，回到主菜单
        scanf("%d", &n);
    }
    return 0;
}

void menu()
{
    system("cls");//清屏操作，清空所有输出
    printf("\n\n\n\n");
    printf("\t\t------------学生信息管理系统------------\n");
    printf("\n");
    printf("\t\t------------1 录入学生信息------------\n");
    printf("\t\t------------2 查找学生信息------------\n");
    printf("\t\t------------3 删除学生信息------------\n");
    printf("\t\t------------4 修改学生信息------------\n");
    printf("\t\t--------------0 退出系统--------------\n");
    printf("\t\t------------请输入0-4：------------\n");
}

void in()//学生录入
{
    char ch[2];//回车也算一个字符
    int m=0;//m表示学生的条数
    FILE* fp;
    fp = fopen("data.txt", "a+");
    if (fp == NULL)
    {
        printf("文件不存在!\n");
        return;
    }
    while (!feof(fp))//循环去查找行数，feof(fp)表示到文件末尾，!feof，没有结束，真，符合条件进入循环
    {
        if (fread(&stu[m], LEN, 1, fp) == 1)//若成功则返回实际读入的count数据，即1
        {
            m++;
        }
    }
    fclose(fp);

    if (m == 0) 
    {
        printf("系统文件中没有学生记录！\n");
    }

    if ((fp = fopen("data.txt", "a+")) == NULL)
    {
        printf("文件不存在！\n");
        return;
    }


    printf("是否录入学生信息(y/n):");
    scanf("%s", ch);
    while (strcmp(ch, "Y") == 0|| strcmp(ch,"y") == 0)//判断是否要输入信息
    {
        printf("number:");
        scanf("%d",&stu[m].num);//输入学生的学号信息
        printf("name:");
        scanf("%s", &stu[m].name);//输入学生的姓名信息
        printf("elec成绩:");
        scanf("%lf", &stu[m].elec);//输入学生的选修课成绩
        printf("expe成绩:");
        scanf("%lf", &stu[m].expe);//输入学生的实验课成绩
        printf("requ成绩:");
        scanf("%lf", &stu[m].requ);//输入学生的必修课成绩
        stu[m].sum = stu[m].elec + stu[m].expe + stu[m].requ;//学生总成绩
        if (fwrite(&stu[m], LEN, 1, fp) != 1)
        {
            printf("无法保存");
        }
        else {
            printf("%s信息被保存成功!",stu[m].name);
            m++;
        }
        printf("是否继续录入？(y/n):\n");
        scanf("%s", ch);
    }
    fclose(fp);
    printf("信息录入结束！\n");
}

void search() //学生信息查找
{
    FILE* fp;
    int snum,i = 0;
    int m = 0;
    fp = fopen("data.txt", "a+");
    if (fp == NULL)
    {
        printf("文件不存在！\n");
        return;
    }
    while (!feof(fp))
    {
        if (fread(&stu[m], LEN, 1, fp) == 1)
        {
            m++;
        }
    }
    fclose(fp);
    if (m == 0)
    {
        printf("文件中没有学生记录\n");
        return;
    }
    printf("请输入学号number查找学生：\n");
    scanf("%d", &snum);
    for (i = 0; i < m; i++)
    {
        if (snum == stu[i].num)
        {
            printf("number     name    elec    esper    requ    sum\n");
            printf(FORMAT, DATA);
            break;
        }
    }
    if (i == m)
    {
        printf("没有找到该学生信息！\n");
    }
}

void del()
{
    int i, j, snum = 0;
    int m = 0;
    char ch[2];
    FILE* fp;
    fp = fopen("data.txt", "a+");
    if (fp == NULL)
    {
        printf("文件不存在！\n");
    }
    while (!feof(fp))
    {
        if (fread(&stu[m], LEN, 1, fp) == 1)
        {
            m++;
        }
    }
    fclose(fp);
    if (m == 0)
    {
        printf("文件中没有学生记录\n");
       
        return;
    }
    printf("请输入删除学生的学号：\n");
    scanf("%d", &snum);
    for (i = 0; i < m; i++)
    {
        if (snum == stu[i].num)
        {
            printf("找到了该学生记录，请问是否确认删除？(y/n)\n");
            scanf("%s", ch);
            if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
            {
                for (int j = i; j < m; j++)
                    stu[j] = stu[j + 1];//将后一个记录移动到前一个记录的位置
                m--;
                printf("删除成功!\n");
            }
            else {
                printf("找到了记录，选择不删除");
            }
            

            if ((fp = fopen("data.txt", "w+")) == NULL)
            {
                printf("文件不存在！\n");
                return;
            }
            for (j = 0; j < m; j++)
            {
                if (fwrite(&stu[j], LEN, 1, fp) != 1)
                {
                    printf("保存失败！\n");
                    
                }
            }
            fclose(fp);
        }
        break;
    }
    if (i == m)
    {
        printf("没有找到该学生信息！\n");
    }
}

void modify()//修改学生信息
{
    FILE* fp;
    int snum,i = 0;
    int m = 0;
    //struct student st;
    if ((fp = fopen("data.txt", "r+")) == NULL)
    {
        printf("文件不存在！\n");
            return;
    }
    while (!feof(fp))
    {
        if (fread(&stu[m], LEN, 1, fp) == 1)
        {
            m++;
        }
    }
    if (m == 0)
    {
        printf("文件中没有信息！\n");
        return;
    }
    printf("请输入要修改的学生的学号number：");
    scanf("%d", &snum);
    for (i = 0; i < m; i++)
    {
        if (snum == stu[i].num)
        {
            printf("找到了该学生，可以修改他的信息！");
            printf("name:");
            scanf("%s", stu[m].name);
            printf("elec:");
            scanf("%lf", &stu[m].elec);
            printf("expe:");
            scanf("%lf", &stu[m].expe);
            printf("requ:");
            scanf("%lf", &stu[m].requ);
            printf("修改成功");
            stu[i].sum = stu[m].elec + stu[m].expe + stu[m].requ;
            if ((fp = fopen("data.txt", "wb")) == NULL)
            {
                printf("文件不存在");
                
                return;
            }
            for (int j = 0; j < m; j++)
            {
                if (fwrite(&stu[j], LEN, 1, fp) != 1)
                {
                    printf("保存失败\n");
                }
            }
            fclose(fp);
            break;
        }
    }
    if (i == m)
    {
        printf("没有找到该学生\n");
    }
}




// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
