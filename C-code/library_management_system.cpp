#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#define LEN sizeof(struct book)
#define len sizeof(struct account)
const int MAX_USERS = 1000;

struct book //确认结构体
{
	int id;
	char name[10];
	char author[10];
	char publish[18];
	double price;
};

struct account
{
	int type;
	char username[10];
	char password[10];
};

struct book shu[50];
struct account acc[50];

void Menu1();//主菜单
void Menu2();//主菜单
void in();//录入
void search();//查找
void delet();//删除
void modify();//修改

void Accmenu();
void accmenu();
void accsearch();
void accdelet();
void accin();

void regist();//注册
void signin();//登录
void Login();//登录界面
void m1();
void m2();

int main()
{
	int n;
	Login();
	scanf("%d", &n);

	while (n)
	{
		switch (n)
		{
		case 1:
			signin(); break;
		case 2:
			regist(); break;

		default:
			break;
		}
		//_getch();
		Login();
		printf("\t\t请再次输入:\t\t\n");
		scanf("%d", &n);
	}

	printf("已退出程序");
	return 0;
}//利用switch函数进行菜单的选择

void Login()
{
	system("cls");
	system("color F3");//界面颜色设置
	printf("\n\n\n\n");
	printf("\t\t**********欢迎使用中传海南国际学院图书管理系统!***********\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~登录界面~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("\n");
	printf("\t\t1.登录账号\t\t\n");
	printf("\t\t2.注册账号\t\t\n");
	printf("\t\t0.退出系统\t\t\n");
	printf("\t\t请输入0-2:\t\t\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void regist()
{
	FILE* fp;
	int m = 0;
	char ch[2];
	fp = fopen("account.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (fread(&acc[m], len, 1, fp) == 1)
	{
		m++;
		if (m >= MAX_USERS) // 假设 MAX_USERS 是用户数组的最大长度
		{
			printf("用户数量已达上限！\n");
			return;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统文件中无用户记录！\n");
	}

	printf("是否注册账号?(y/n):");
	scanf("%s", ch);

	if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
	{
		if ((fp = fopen("account.txt", "a+")) == NULL)
		{
			printf("文件不存在！\n");
			return;
		}

		printf("请输入用户类型(1.管理员/2.普通用户):");
		scanf("%d", &acc[m].type);
		printf("请输入账号名:");
		scanf("%19s", acc[m].username); // 限制输入长度
		printf("请输入账号密码:");
		scanf("%19s", acc[m].password); // 限制输入长度

		if (fwrite(&acc[m], len, 1, fp) != 1)
		{
			printf("无法保存");
		}
		else
		{
			printf("%s用户信息被保存成功！", acc[m].username);
		}

		fclose(fp);
	}
}

void signin()
{
	FILE* fp;
	int m = 0, i = 0;
	char ch[2];
	char username[10], password[10];

	fp = fopen("account.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (!feof(fp))
	{
		if (fread(&acc[m], len, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统文件中无用户记录！请先前往注册(填n返回主界面注册)！\n");
	}

	if ((fp = fopen("account.txt", "a+")) == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	printf("是否登录账号?(y/n):");
	scanf("%s", ch);

	if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
	{
		printf("请输入账号名:");
		scanf("%10s", username);
		printf("请输入账号密码:");
		scanf("%10s", password);

		for (i = 0; i < m; i++)
		{
			if (strcmp(username, acc[i].username) == 0 && strcmp(password, acc[i].password) == 0)
			{
				printf("登录成功!\n");
				printf("是否进入主菜单？（y/n):\n");
				scanf("%s", ch);

				if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
				{
					if (acc[i].type == 1) {
						m1();
					}
					else {
						m2();
					}
				}
			}
		}
	}
	if (i == m)
	{
		printf("没有找到该账户,登录失败！\n");
	}
}

void m1() {
	int n;
	Menu1();
	scanf("%d", &n);

	while (n)
	{
		switch (n)
		{
		case 1:in(); break;
		case 2:search(); break;
		case 3:delet(); break;
		case 4:modify(); break;
		case 5:accmenu(); break;
		default: break;
		}
		_getch();
		Menu1();
		scanf("%d", &n);
	}

}

void Menu1() //管理员系统
{
	system("cls");
	system("color F3");//界面颜色设置
	printf("\n\n\n\n");
	printf("\t\t**********欢迎使用中传海南国际学院图书管理系统!***********\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~主菜单(管理员)~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("\n");
	printf("\t\t1.录入图书\t\t\n");
	printf("\t\t2.查找图书\t\t\n");
	printf("\t\t3.删除图书\t\t\n");
	printf("\t\t4.修改图书\t\t\n");
	printf("\t\t5.用户信息管理\t\t\n");
	printf("\t\t0.返回登录界面\t\t\n");
	printf("\t\t请输入0-5:\t\t\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void m2() {
	int n;
	Menu2();
	scanf("%d", &n);

	while (n)
	{
		switch (n)
		{
		case 1:search(); break;
		default:

			break;
		}
		_getch();
		Menu2();
		scanf("%d", &n);
	}
	printf("即将返回上级页面，请按任意键继续……");
}

void Menu2()//普通用户
{
	system("cls");
	system("color F3");//界面颜色设置
	printf("\n\n\n\n");
	printf("\t\t**********欢迎使用中传海南国际学院图书管理系统!***********\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~主菜单(普通用户)~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("\n");
	printf("\t\t1.查找图书\t\t\n");
	printf("\t\t0.返回登录界面\t\t\n");
	printf("\t\t请输入( 0 / 1 ):\t\t\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void in()//录入图书
{
	FILE* fp;
	int m = 0;//m表示书籍条数
	char ch[2];

	fp = fopen("book.txt", "a+");//打开文件

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (!feof(fp))
	{
		if (fread(&shu[m], LEN, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统文件中无书籍记录！\n");
	}
	if ((fp = fopen("book.txt", "a+")) == NULL)
	{
		printf("文件不存在！\n");

		return;
	}

	printf("是否录入书籍信息?(y/n):");
	scanf("%s", ch);

	while (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
	{
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		printf("请输入图书id录入图书：\n");
		scanf("%d", &shu[m].id);
		printf("请输入书名：\n");
		scanf("%s", shu[m].name);
		printf("请输入书籍作者：\n");
		scanf("%s", shu[m].author);
		printf("请输入书籍出版社：\n");
		scanf("%s", shu[m].publish);
		printf("请输入书籍价格：\n");
		scanf("%lf", &shu[m].price);
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

		if (fwrite(&shu[m], LEN, 1, fp) != 1)
		{
			printf("无法保存");
		}
		else
		{
			printf("%s信息被保存成功！", shu[m].name);
			m++;
		}
		printf("是否继续录入书籍？（y/n):\n");
		scanf("%s", ch);
	}

	fclose(fp);
	printf("信息录入结束！\n");
}

void search()
{
	FILE* fp;
	int iid;
	int i = 0;
	int m = 0;
	fp = fopen("book.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (!feof(fp))
	{
		if (fread(&shu[m], LEN, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("文件中没有书籍记录！\n");
		return;
	}

	printf("请输入书籍id查找书籍：\n");
	scanf("%d", &iid);

	for (i = 0; i < m; i++)
	{
		if (iid == shu[i].id)
		{
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
			printf("        id           name          author                 publish             price\n");
			printf("%-8d%-8s%-6s%-6s%-8.8lf\n", shu[i].id, shu[i].name, shu[i].author, shu[i].publish, shu[i].price);
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
			break;
		}
	}
	if (i == m)
	{
		printf("没有找到该书籍信息！\n");
	}
}

void delet()
{
	int iid;
	int i = 0;
	int j = 0;
	int m = 0;
	char ch[2];
	FILE* fp;
	fp = fopen("book.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
	}

	while (!feof(fp))
	{
		if (fread(&shu[m], LEN, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);
	if (m == 0)
	{
		printf("文件中没有书籍记录！\n");
		return;
	}

	printf("请输入删除的书籍id:\n");
	scanf("%d", &iid);

	int flag = -1;

	for (i = 0; i < m; i++)
	{
		if (iid == shu[i].id)
		{
			flag *= -1;

			printf("找到了书籍记录，请问是否删除！(y/n):\n");
			scanf("%s", ch);

			if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
			{
				for (int j = i; j < m; j++)
					shu[j] = shu[j + 1];//覆盖
				m--;
				printf("删除成功!\n");
			}
			else {
				printf("找到了记录，选择不删除");
			}

			if ((fp = fopen("book.txt", "w+")) == NULL)
			{
				printf("文件不存在！\n");

				return;
			}

			for (j = 0; j < m; j++)
			{
				if (fwrite(&shu[j], LEN, 1, fp) != 1)
				{
					printf("保存失败！\n");
				}
			}
			fclose(fp);
		}
		break;
	}
	if (flag == -1)
	{
		printf("没有找到该书籍信息！\n");
	}
}

void modify()
{
	FILE* fp;
	int  iid;
	int i = 0;
	int m = 0;

	fp = fopen("book.txt", "r+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (fread(&shu[m], sizeof(struct book), 1, fp) == 1)
	{
		m++;
	}

	if (m == 0)
	{
		printf("文件中没有信息！\n");
		fclose(fp); // 关闭文件
		return;
	}

	printf("请输入要修改的书籍的id：");
	scanf("%d", &iid);

	for (i = 0; i < m; i++)
	{

		if (iid == shu[i].id)
		{
			printf("找到了该书籍，可以修改它的信息！\n");
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
			printf("书籍id:");
			scanf("%d", &shu[i].id);
			printf("书名:");
			scanf("%s", shu[i].name);
			printf("书籍作者:");
			scanf("%s", shu[i].author);
			printf("书籍出版社:");
			scanf("%s", shu[i].publish);
			printf("书籍价格:");
			scanf("%lf", &shu[i].price);
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

			printf("修改成功\n");

			fseek(fp, 0, SEEK_SET); // 将文件指针定位到文件开头

			for (int j = 0; j < m; j++)
			{
				if (fwrite(&shu[j], len, 1, fp) != 1)
				{
					printf("保存失败\n");
				}
			}
			fclose(fp); // 关闭文件
			return;
		}
	}
	printf("没有找到该书籍\n");

	fclose(fp); // 关闭文件
}

void accmenu()
{
	int n;
	Accmenu();
	scanf("%d", &n);
	while (n)
	{
		switch (n)
		{
		case 1:accin(); break;
		case 2:accsearch(); break;
		case 3:accdelet(); break;
		default:

			break;
		}
		_getch();
		Accmenu();
		scanf("%d", &n);
	}

	printf("即将返回上级页面，请按任意键继续……");
}

void Accmenu()//用户信息管理系统
{
	system("cls");
	system("color F3");//界面颜色设置
	printf("\n\n\n\n");
	printf("\t\t**********欢迎使用中传海南国际学院用户信息管理系统!***********\n");
	printf("\n");
	printf("\t\t1.录入用户信息\t\t\n");
	printf("\t\t2.查找用户信息\t\t\n");
	printf("\t\t3.删除用户信息\t\t\n");
	printf("\t\t0.返回管理员操作界面\t\t\n");
	printf("\t\t请输入0-3:\t\t\n");
	printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void accsearch()
{
	FILE* fp;
	char username[10];
	int i = 0, m = 0;
	fp = fopen("account.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		return;
	}

	while (!feof(fp))
	{
		if (fread(&acc[m], len, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统中没有用户记录！\n");
		return;
	}

	printf("请输入用户名查找用户：\n");
	scanf("%s", &username);

	for (i = 0; i < m; i++)
	{
		if (strcmp(username, acc[i].username) == 0)
		{
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
			printf("    username    password     \n");
			printf("%-8s%-10s%\n", acc[i].username, acc[i].password);
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
			break;
		}
	}

	if (i == m)
	{
		printf("没有找到该用户信息！\n");
	}
}

void accdelet()
{
	FILE* fp;
	char username[10];
	int i = 0, j = 0, m = 0;
	char ch[2];

	fp = fopen("account.txt", "a+");

	if (fp == NULL)
	{
		printf("文件不存在！\n");
	}

	while (!feof(fp))
	{
		if (fread(&acc[m], len, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统中没有用户记录！\n");
		return;
	}

	printf("请输入用户名查找待删除用户：\n");
	scanf("%s", username);

	int flag = -1;

	for (i = 0; i < m; i++)
	{
		if (strcmp(username, acc[i].username) == 0)
		{

			flag *= -1;

			printf("找到了该用户账户，请问是否删除！(y/n):\n");
			scanf("%s", ch);

			if (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
			{
				for (int j = i; j < m; j++)
					acc[j] = acc[j + 1];//覆盖
				m--;
				printf("删除成功!\n");
			}
			else {
				printf("找到了该用户账户信息，选择不删除");
			}

			if ((fp = fopen("account.txt", "w")) == NULL)
			{
				printf("文件不存在！\n");
				return;
			}
			for (j = 0; j < m; j++)
			{
				if (fwrite(&acc[j], LEN, 1, fp) != 1)
				{
					printf("保存失败！\n");
				}
			}
			fclose(fp);
			break;
		}
	}
	if (flag == -1)
	{
		printf("没有找到该用户账户信息！\n");
	}
}

void accin()//录入用户信息
{
	FILE* fp;
	int m = 0;//m表示条数
	char ch[2];
	fp = fopen("account.txt", "a+");//打开文件

	if (fp == NULL)
	{
		printf("文件不存在！\n");
		printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		return;
	}

	while (!feof(fp))
	{
		if (fread(&acc[m], len, 1, fp) == 1)
		{
			m++;
		}
	}

	fclose(fp);

	if (m == 0)
	{
		printf("系统文件中无用户信息记录！\n");
	}

	if ((fp = fopen("account.txt", "a+")) == NULL)
	{
		printf("文件不存在！\n");
		printf("\t\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

		return;
	}

	printf("是否录入用户信息?(y/n):");
	scanf("%s", ch);

	while (strcmp(ch, "Y") == 0 || strcmp(ch, "y") == 0)
	{
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		printf("请输入用户类型(1.管理员/2.普通用户):");
		scanf("%d", &acc[m].type);
		printf("请输入账号名:");
		scanf("%19s", acc[m].username); // 限制输入长度
		printf("请输入账号密码:");
		scanf("%19s", acc[m].password); // 限制输入长度
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

		if (fwrite(&acc[m], len, 1, fp) != 1)
		{
			printf("无法保存");
		}

		else
		{
			printf("%s信息被保存成功！", acc[m].username);
			m++;
		}

		printf("是否继续录入用户信息？（y/n):\n");
		scanf("%s", ch);
	}

	fclose(fp);

	printf("用户信息录入结束！\n");
}

