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
        l,r = fenzu(left)[1],fenzu(left)[2]
        while len(l) > 1 or len(r) > 1:
            left = fenzu(left)[0]
            l,r = fenzu(left)[1],fenzu(left)[2]
            print(left,l,r)
    if len(right) > 1:
        right = fenzu(right)[0]
        l,r = fenzu(right)[1],fenzu(right)[2]
        while len(l) > 1 or len(r) > 1:
            right = fenzu(right)[0]
            l,r = fenzu(right)[1],fenzu(right)[2]
            print(right,l,r)

    sequence = left + middle + right
    print(sequence,left,right,middle)
    return sequence

print(quickSort([-100, -18, 53, -45, -66, -17, -9, -71, 49, -75]))