def _binarySearch( sequence, value ):
    if value < sequence[len(sequence)//2]:
        sequence = sequence[:len(sequence)//2]
        if len(sequence) > 0:
            binarySearch(sequence,value)
        else:
            return False
    elif value > sequence[len(sequence)//2]:
        sequence = sequence[len(sequence)//2:]
        if len(sequence) > 0:
            binarySearch(sequence,value)
        else:
            return False
    else:
        return True


def __binarySearch(sequence, value):#chatgpt修改,主要是由于递归调用时未正确处理递归返回值。此外，还可以进行一些优化和简化
    if not sequence:
        return False  # 空序列，未找到目标值

    mid = len(sequence) // 2

    if value < sequence[mid]:
        return binarySearch(sequence[:mid], value)
    elif value > sequence[mid]:
        return binarySearch(sequence[mid + 1:], value)
    else:
        return True  # 找到目标值
#这里的修改主要包括以下几点：
#添加了对空序列的检查，如果序列为空，则直接返回 False。
#将 sequence 切片操作替换为对应的参数传递，避免修改原始序列，从而确保每次递归调用都在新的子序列上进行。
#在递归调用的返回值上进行正确的处理，确保递归调用的结果正确地传播到最外层。

def binarySearch(sequence, value, left=0, right=None):
    if right is None:
        right = len(sequence) - 1

    if left <= right:
        mid = (left + right) // 2

        if sequence[mid] == value:
            return True  # 找到了目标值
        elif sequence[mid] < value:
            return binarySearch(sequence, value, mid + 1, right)
        else:
            return binarySearch(sequence, value, left, mid - 1)

    return False  # 目标值未找到


# 示例用法:
sorted_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
search_value = 7

result = binarySearch(sorted_sequence, search_value)

if result:
    print(f"{search_value} 在序列中找到了。")
else:
    print(f"{search_value} 在序列中未找到。")

#print(binarySearch([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96],0))