def fast_power(a, n):
    if n == 0:
        return 1
    if n == 1:
        return a
    if n % 2 == 1:
        return fast_power(a, n - 1) * a
    else:
        return fast_power(a, n / 2) * fast_power(a, n / 2)


print(fast_power(2, 1099))
