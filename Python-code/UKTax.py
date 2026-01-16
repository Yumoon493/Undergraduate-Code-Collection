def personalAllowance(n):
    if 0 <= n <= 100000 :
        return 12500
    elif n > 100000:
        f = 12500 - (n-100000)//2
        if f >= 0:
          return f
        else:
          return 0
    else:
        return 0

def incomeTax(n):
    ts = n - personalAllowance(n)
    if n < personalAllowance(n):
        tax = 0
    else:
        if 0<=ts<=37500:
            tax = ts*0.2
        elif 37500<ts<=150000:
            tax = 37500*0.2 + (ts-37500)*0.4
        else:
            tax = 37500*0.2 + 0.4*(150000-37500) + 0.45*(ts-150000)
    return tax

def monthlyPay(n):
  pAm = int(personalAllowance(n)/12)
  m = int((n - incomeTax(n))/12)
  return pAm,m

def StudentLoan(n,boolean):
    if boolean == True:
        if n>25725:
            r = int(0.09*(monthlyPay(n)[0]-25725))
        else:
            r = 0
        return r
    else:
        return 0

monthlyPay()
StudentLoan()