def binarySearch( sequence, value ):
    for i in range(len(sequence)):
        if value == sequence[i]:
            return True
    return False

binarySearch([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96], 20)