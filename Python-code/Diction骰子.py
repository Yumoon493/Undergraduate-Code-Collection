def dictPrinter(inD):

    # First we find the longest key
    m = 0
    for var in inD:
        size = len(str(var))
        if size > m:
            m = size

    # Then we print out pairs so colons align
    for var in inD:
        keyStr = str(var)
        size = len(keyStr)
        print(keyStr + " " * (m - size) + " : " + str(inD[var]))
    print("")


# stock = {'bananas': 31, 'oranges': 55, 'apples': 40, "pears":33}
# dictPrinter(stock)

# eng2sp = { "one":"uno", "two":"dos", "three":"tres" }
# dictPrinter(eng2sp)

diceDict = {}
for dice1 in range(1,7):
    for dice2 in range(1,7):
        for dice3 in range(1,7):
            tot = dice1+dice2+dice3
            if tot in diceDict:
                diceDict[tot].append( (dice1,dice2,dice3) )
            else:
                diceDict[tot] = [(dice1,dice2,dice3)]

dictPrinter(diceDict)