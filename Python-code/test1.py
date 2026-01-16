"""
Question1: pseudocode
input : string s , character c
output : int n of the occurrence of c in s
n = 0
s = s.lower()
for each character i in s:
    if i == c:
        n += 1
return n
"""
""""
s = input("Please give me a sentence:")
c = input("Choose the key character in the sentence:")
n = 0
s = s.lower()
for i in s :
    if i == c :
        n = n+1
output = f"Occurrence of {c} in the sentence is {n}"
print(output)
"""

#(3)
s1 = input("a string to reverse:")
new_s = ""
def rev(s1):
    if len(s1) <= 1:
        return s1
    return s1[-1] + rev(s1[0:-1])

print(rev(s1))

#for i in s1:
#    new_s =  i + new_s

#l = []
# for i in s1:
#    l.append(i)