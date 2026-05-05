from __future__ import annotations

from genetics import (
    AAllele,
    BAllele,
    DAllele,
    Genotype,
    OAllele,
    Sex,
    WAllele,
    enumerate_genotypes,
)


def genotype_to_phenotype(g: Genotype) -> str:
    # Dominant white masks everything
    if g.W_locus[0] == WAllele.W:
        return "White"

    is_dilute = g.D_locus == (DAllele.d, DAllele.d)
    is_tabby = g.A_locus[0] == AAllele.A

    # O locus: hemizygous in males, diploid in females
    if g.sex == Sex.MALE:
        is_orange = g.O_locus[0] == OAllele.XO
        is_tortie = False
    else:
        o1, o2 = g.O_locus  # canonical: XO before Xo
        is_orange = o1 == OAllele.XO and o2 == OAllele.XO
        is_tortie = o1 == OAllele.XO and o2 == OAllele.Xo

    if is_orange:
        base = "Cream" if is_dilute else "Orange"
        return (base + " tabby") if is_tabby else base

    if is_tortie:
        if is_dilute:
            return "Blue-cream tabby" if is_tabby else "Blue-cream"
        return "Torbie" if is_tabby else "Tortoiseshell"

    # Non-orange: B locus determines base color (canonical first allele is dominant)
    b1 = g.B_locus[0]
    if b1 == BAllele.B:
        base = "Blue" if is_dilute else "Black"
    elif b1 == BAllele.bprime:  # only reachable for b'b' since b' never dominates b
        base = "Fawn" if is_dilute else "Cinnamon"
    else:  # b1 == b: covers bb and bb'
        base = "Lilac" if is_dilute else "Chocolate"

    return (base + " tabby") if is_tabby else base


def get_valid_phenotypes(sex: Sex) -> list[str]:
    return sorted({genotype_to_phenotype(g) for g in enumerate_genotypes(sex)})


MALE_PHENOTYPES: list[str] = get_valid_phenotypes(Sex.MALE)
FEMALE_PHENOTYPES: list[str] = get_valid_phenotypes(Sex.FEMALE)
