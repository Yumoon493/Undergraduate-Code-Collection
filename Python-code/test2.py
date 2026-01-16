#Bubble sort (a)用时22min
def bubble(s):
    swap = True
    comparisons = 0
    swaps = 0
    roundnum = 0
    while swap is True:
       swap = False
       for i in range(len(s)-1-roundnum):
           if roundnum <= 2:
               comparisons += 1
           if s[i] > s[i+1]:
               swap = True
               p = s[i+1]
               s[i+1] = s[i]
               s[i] = p
               if roundnum <= 2:
                   swaps += 1
       roundnum += 1
    print(f"Comparisons are {comparisons}, and swaps are {swaps}.")
    return s

s = [13,16,10,11,4,12,6,7]
bubble(s)
print(s)

"""
(b)
1.insertion sort 
2.heap sort
3.insertion sort

(c)
For Quick sort:
best: O(n log n)) when pivot is chosen balanced (two parts divided by pivot are nearly close)
worst: O(n*n) when pivot is chosen unbalanced (pivot close to two sides)

"""

"""
(d)
 Solves the '8 queens' problem for n queens, producing all
    solutions and returning the total number of solutions. """

from datetime import datetime
def is_available(n, table, column, N):
    return not any(t in (n, n - i, n + i) for t, \
        i in zip(table, range(column, 0, -1)))

def queens_sum(N):
    return solve(N, [0] * N, 0, N)

def solve(N, table, column, end):
    if column == end:
        print('table is: ', table)
        return 1
    summ = 0
    for n in range(N):
        if is_available(n, table, column, N):
            table[column] = n
            summ = summ + solve(N, table, column + 1, end)
    return summ
start = datetime.now()

N = 8
solutions = queens_sum(N)

end =  datetime.now()

total_time = end - start

print(f"The total number of solutions is {solutions}")
print(f"The total time taken was {total_time}")


