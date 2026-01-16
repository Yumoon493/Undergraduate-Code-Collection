

def _compare(filename1, filename2):
    f = open(filename1,"r")
    p = open(filename2,"r")
    l = min(len(f.readlines()),len(p.readlines()))
    for line in range(l):
        if f.readlines()[line] != p.readlines()[line]:
            return line+1
        if line == len(f)-1 and f[line] == p[line]:
            return 0
    f.close()
    p.close()


def compare(filename1, filename2):
    with open(filename1, "r", encoding='utf-8') as f, open(filename2, "r", encoding='utf-8') as p:
        lines_f = f.readlines()
        lines_p = p.readlines()
        l = min(len(lines_f), len(lines_p))

        for line in range(l):
            if lines_f[line] != lines_p[line]:
                return line + 1

        # Check for remaining lines in longer file
        if len(lines_f) != len(lines_p):
            return min(len(lines_f), len(lines_p)) + 1

    return 0


print(compare(r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\Python\txt使用\BoringTextFile.txt",r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\Python\txt使用\DifferentBoringTextFile-SameLength.txt"))