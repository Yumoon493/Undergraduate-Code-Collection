def selection_sort(A):
    for i in range(len(A) - 1):        				# Loop over each element in the array, except the last one
        min_index = i   									# Assume the current index holds the minimum value
        for j in range(i + 1, len(A)):   			# Check the rest of the array to find the true minimum element
            if A[j] < A[min_index]:
                min_index = j    						# Update min_index if a smaller element is found
        A[i], A[min_index] = A[min_index], A[i]	# Swap the smallest element found with the element at index i

    return A   # Return the sorted array

A = [11, 22, 14, 67, 2, 9]   							# Example usage
sorted_array = selection_sort(A)
print("Sorted array:", sorted_array)
