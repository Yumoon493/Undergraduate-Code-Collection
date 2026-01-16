def fenzu(sequence):
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
    return sequence, l, r, m

def fenzu(sequence):
    l = []
    r = []
    m = []
    for i in range(len(sequence)):
        p = sequence[(len(sequence) - 1) // 2]
        if sequence[i] < p:
            l.append(sequence[i])
        elif sequence[i] > p:
            r.append(sequence[i])
        else:
            m.append(p)
    sequence = l + m + r
    return sequence,l,r,m

def quickSort( sequence ):
    fenzu(sequence)
    sequence,left,right,middle = fenzu(sequence)
    if len(left) > 1:
        left = fenzu(left)[0]
    if len(right) > 1:
        right = fenzu(right)[0]

    sequence = left + middle + right
    return sequence

