def MergeStrings(s1, s2):
    s3 = ""  # Define s3 inside the function

    # Determine which string is shorter
    if len(s1) > len(s2):
        for i in range(len(s2)):
            s3 += s1[i] + s2[i]
        s3 += s1[len(s2):]
    else:
        for i in range(len(s1)):
            s3 += s1[i] + s2[i]
        s3 += s2[len(s1):]

    print(s3)


#s1 = input("The first string is: ")
#s2 = input("The second string is (one string is longer than the other): ")
#MergeStrings(s1, s2)

import math
def descriminative(num):
    i = 1
    while num/i > 10:
        i = i*10
    n = math.log10(i)+1
    return n
#print(descriminative(333))


def fac(n):
    while n > 1:
        return n*fac(n-1)
    return 1
#print(fac(3))
def dev(n,m):
    x = fac(n)
    if x%m == 0:
        return True
    else:
        return False
print(f"is {dev(6,9)}")





