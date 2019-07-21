def upper_bound(d, g):
    return 4 / (d - 2) * (d - 1) ** (g + 2)


def lower(d, g):
    assert g % 2 == 0
    assert d > 1
    if d == 2:
        return g / 2
    if g == 2:
        return 1
    if g == 4:
        return 2
    if g == 6:
        return d + 1
    if g == 8:
        return 2 * d
    count = 1
    if ((g / 2) % 2) == 1:
        for i in range((g - 6) / 4):
            count += d * (d - 1) ** i
        return count
    else:
        for i in range((g - 8) / 4):
            count += d * (d - 1) ** i + (d - 1) ** ((g - 4) / 4)
        return count
