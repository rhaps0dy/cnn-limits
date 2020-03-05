
def p(mu, nu, Ih=4, Id=5, Ph=2, Pd=3, Sh=1, Sd=1):
    patches_wide = (Id - Pd + 1) // Sd
    return (
        (nu % Pd) + (mu%patches_wide) *Sd
        + (nu//Pd + (mu//patches_wide) *Sh) * Ih)
