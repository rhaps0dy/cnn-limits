import numpy as np


def gen_patches(Fh, Fw, Sh, Sw, Ih, Iw):
    img = np.arange(Ih*Iw).reshape((Ih, Iw))

    patches = []
    for i in range(0, Ih-Fh+1, Sh):
        for j in range(0, Iw-Fw+1, Sw):
            patches.append(img[i:i+Fh, j:j+Fw].ravel())
    return patches


def p(Fh, Fw, Sh, Sw, _Ih, Iw, Ih_l1, Iw_l1, mu, i):
    assert i < Fh*Fw, f"invalid index i: {i}"
    assert mu < Ih_l1*Iw_l1, f"invalid index mu: {mu}"
    return ((i % Fw) + Sw*(mu % Iw_l1) + Iw*(
        i//Fw + Sh*(mu//Iw_l1)))


def test():
    for Fh in range(1, 5+1):
        for Fw in range(1, 5+1):
            for Sh in [1, 2]:
                for Sw in [1, 2]:
                    for Ih in range(2, 8):
                        for Iw in range(2, 8):

                            Iw_l1 = (Iw - Fw) // Sw + 1
                            Ih_l1 = (Ih - Fh) // Sh + 1
                            if Iw_l1 <= 0 or Ih_l1 <= 0:
                                continue

                            gen = gen_patches(Fh, Fw, Sh, Sw, Ih, Iw)
                            g2 = [[p(Fh, Fw, Sh, Sw, Ih, Iw, Ih_l1, Iw_l1, mu, i)
                                for i in range(Fw*Fh)] for mu in range(Iw_l1*Ih_l1)]
                            if not np.array_equal(gen, g2):
                                print(f"Fh={Fh}; Fw={Fw}; Sh={Sh}; Sw={Sw}; Ih={Ih}; Iw={Iw}")
                                return



def lemma1():
    Fh=Fw = 3
    Sh=Sw = 2
    Ih= 6
    Iw = 7
    Iw_l1 = (Iw - Fw) // Sw + 1
    Ih_l1 = (Ih - Fh) // Sh + 1

    p_ = lambda mu, i: p(Fh, Fw, Sh, Sw, Ih, Iw, Ih_l1, Iw_l1, mu, i)

    for mu in range(Iw_l1*Ih_l1):
        for mu2 in range(Iw_l1*Ih_l1):
            d = p_(mu, 0) - p_(mu2, 0)
            for i in range(1, Fh*Fw):
                assert p_(mu, i) - p_(mu2, i) == d

def lemma2():
    Fh=Fw = 3
    Sh=Sw = 2
    Ih= 6
    Iw = 7
    Iw_l1 = (Iw - Fw) // Sw + 1
    Ih_l1 = (Ih - Fh) // Sh + 1

    p_ = lambda mu, i: p(Fh, Fw, Sh, Sw, Ih, Iw, Ih_l1, Iw_l1, mu, i)

    for mu in range(Iw_l1*Ih_l1):
        for mu2 in range(Iw_l1*Ih_l1):
            d = p_(mu, 0) - p_(mu2, 0)


a = np.zeros((9, 9), int)
b = np.zeros((25, 25), int)
patches_a = gen_patches(2, 2, 1, 1, 3, 3)
patches_b = gen_patches(3, 3, 1, 1, 5, 5)

def g(k, l):
    b[patches_b[k], patches_b[l]] += 1
    print("   1 2 3 4 5 6 7 8 9 0.1.2.3.4.5.6.7.8.9.0 1 2 3 4 5\n"
          + "\n".join(f"{i+1:02d}"
                      + "".join((f" {c:01d}" if c > 0 else "  ")
                                for c in row)
                      for i, row in enumerate(b)))

def h():
    print("   1 2 3 4 5 6 7 8 9\n"
          + "\n".join(f"{i+1:02d}"
                      + "".join((f" {c:01d}" if c > 0 else "  ")
                                for c in row)
                      for i, row in enumerate(a)))

a[1, 3] = 1  # or whatever

for i in range(9):
    for j in range(9):
        if a[i, j] > 0:
            g(i, j)

for i in range(4):
    for j in range(4):
        a[patches_a[i], patches_a[j]] += 1

np.sum([(np.diag(a, i) > 0).any() for i in range(9)])
