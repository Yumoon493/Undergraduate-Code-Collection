def mergeSort(sequence):
    if len(sequence) <= 1:
        return sequence

    # Split the sequence into two halves
    mid = len(sequence) // 2
    left_half = sequence[:mid]
    right_half = sequence[mid:]

    # Recursively sort each half
    left_half = mergeSort(left_half)
    right_half = mergeSort(right_half)

    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left, right):
    result = []
    left_index = right_index = 0

    # Compare elements from the left and right halves and merge
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1

    # Append remaining elements from both halves (if any)
    result.extend(left[left_index:])
    result.extend(right[right_index:])

    return result

# Example usage:
unsorted_sequence = [38, 27, 43, 3, 9, 82, 10]
sorted_sequence = mergeSort(unsorted_sequence)
print("Original Sequence:", unsorted_sequence)
print("Sorted Sequence:", sorted_sequence)
