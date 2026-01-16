def __bubbleSort( sequence ):
    for i in range(len(sequence)):
        if i == 0:
            continue
        if sequence[i-1] > sequence[i]:
            sequence[i],sequence[i-1] = sequence[i-1],sequence[i]
            print(sequence)
            bubbleSort(sequence)
    return sequence
#print(bubbleSort([-98, 84, -87, -29, -69, -46, -76, 27, 24, 1]))
#你的代码中有一个小问题，就是在每次交换元素后都调用了 bubbleSort(sequence)，这样会导致递归深度增加。通常，递归调用应该在整个循环完成之后进行。
def _bubbleSort(sequence):
    for i in range(len(sequence)):
        if i == 0:
            continue
        if sequence[i - 1] > sequence[i]:
            sequence[i], sequence[i - 1] = sequence[i - 1], sequence[i]
            print(sequence)

    # 在循环结束后，递归调用
    if len(sequence) > 1:
        bubbleSort(sequence[1:])

    return sequence

def bubbleSort(sequence):
    '''Input is a list, output a list with
    the same data sorted using Bubble Sort.'''

    n = len(sequence)

    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already sorted, so we don't need to check them
        for j in range(0, n - i - 1):
            # Swap if the element found is greater than the next element
            if sequence[j] > sequence[j + 1]:
                sequence[j], sequence[j + 1] = sequence[j + 1], sequence[j]

    return sequence


# Example usage:
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = bubbleSort(my_list)
print("Sorted list:", sorted_list)

#use recursive
def recursiveBubbleSort(sequence, n=None):
    '''Input is a list, output a list with
    the same data sorted using Recursive Bubble Sort.'''

    # If n is not provided, set it to the length of the sequence
    if n is None:
        n = len(sequence)

    # Base case: If there's only one element or the list is empty, it's already sorted
    if n <= 1:
        return sequence

    # Traverse through all array elements
    for i in range(n - 1):
        # Swap if the element found is greater than the next element
        if sequence[i] > sequence[i + 1]:
            sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]

    # Recursively call the function with the reduced length of the sequence
    return recursiveBubbleSort(sequence, n - 1)


# Example usage:
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = recursiveBubbleSort(my_list)
print("Sorted list:", sorted_list)


