s1 = input("The first string is:")
s2 = input("The second string is(one string is longer than the other):")
s3 = ""
def MergeStrings(s1,s2):

    if len(s1) > len(s2):
        for i in range(len(s2)-1):
            s = s1[i] + s2[i]
            s3 = s3 + s
            i = i + 1

        s4 = s3 + s1[len(s2)-1:]
        print(s4)

    if len(s1) < len(s2):
        for i in range(len(s1)-1):
            s = s2[i] + s1[i]
            s3 = s3 + s
            i = i + 1

        s4 = s3 + s2[len(s1) - 1:]
        print(s4)

MergeStrings(s1,s2)

