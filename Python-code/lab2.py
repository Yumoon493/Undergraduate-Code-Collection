#Non-assessed Task 1: Palindrome
#Non-assessed Task 2: Bubble Sort implementation

def bubblesort1(list):
    for i in range(len(list)-1):
        if list[i] <= list[i+1]:
            continue
        else:
            list[i],list[i+1] = list[i+1],list[i]
    return list


print(bubblesort1([1,3,2,5,4]))
#NONE...(wrong)
#Forget return, outside the loop.



#Non-assessed Task 3: Merge Sort implementation

def mergesort(l):
    if len(l) <= 1:
        return l
    p = len(l)//2
    a = l[:p]
    b = l[p:]
    a = mergesort(a)
    b = mergesort(b)

    return merge(a,b)

def merge(a, b):
    merged_list = [0] * (len(a) + len(b))  # print('lefthalf in merge is:，lefthalf)
    i = 0
    j = 0
    k = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged_list[k] = a[i]
    # if element in left is less than or equal#add it to the merged list
    #move to the next item in the left list
            i = i + 1
        else:
            merged_list[k] = b[j]
    # otherwise move the element in the right
    #to the merged list and move to the next
    #move to the next item in the merged list
            j = j + 1
        k = k + 1
    # print('merged list after sort loop is:',merged list)
    #print('lefthalf after sort loop is:lefthalf)
    #print('righthalf after sort loop is:，righthalf)
    while i < len(a):
        merged_list[k] = a[i]
        i = i + 1
        k = k + 1

    while i < len(b):
        merged_list[k] = b[j]
        j = j + 1
        k = k + 1

    # if items are left over in the left half,#add to the merged list
    # if items are left over in the right half#add to the merged list
    # print("Merging ",merged list)

    return merged_list

print(mergesort([1,3,2,5,4]))