f = open(r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\Python\words.txt", "r")
oneGiantString = f.read()
f.close()

wordList = oneGiantString.split()#Finish This Line!

p = open(r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\Python\words2.txt","w")
for word in wordList:
    if word[-1] == 's':
        word = word[0:-2] + "\n"
        p.write(word)
    else:
        word = word + "\n"
        p.write(word)
p.close()
