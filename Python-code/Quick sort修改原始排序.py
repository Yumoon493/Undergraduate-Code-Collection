def quickSort(sequence):
    if len(sequence) <= 1:
        return sequence, [], [], []
    l = []
    r = []
    m = []
    p = sequence[(len(sequence) - 1) // 2]
    for i in range(len(sequence)):
        if sequence[i] < p:
            l.append(sequence[i])
        elif sequence[i] > p:
            r.append(sequence[i])
        else:
            m.append(p)
    sequence = l + m + r
    if len(l) > 1:
        l = quickSort(l)

    if len(r) > 1:
        r = quickSort(r)

    sequence = l + m + r
    return sequence

print(quickSort([-10, -90, -55, 40, 29, -96, -30, 14, -98, -37]))

