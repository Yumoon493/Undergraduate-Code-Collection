def quickSort(sequence):
    '''Input is a list, output a list with
    the same data sorted using Quick Sort.'''

    if len(sequence) <= 1:
        return sequence
    else:
        pivot_index = len(sequence) // 2
        pivot = sequence[pivot_index]

        less = []
        greater = []

        # Loop through the elements before the pivot
        for x in sequence[:pivot_index]:
            if x <= pivot:
                less.append(x)
            else:
                greater.append(x)

        # Loop through the elements after the pivot
        for x in sequence[pivot_index + 1:]:
            if x <= pivot:
                less.append(x)
            else:
                greater.append(x)

        return quickSort(less) + [pivot] + quickSort(greater)
    #这行最重要（上）


# Example usage:
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_list = quickSort(my_list)
print(sorted_list)



 