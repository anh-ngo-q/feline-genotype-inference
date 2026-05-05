from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations_with_replacement, product as iproduct
from typing import Iterator, Tuple


class BAllele(Enum):
    B = "B"
    b = "b"
    bprime = "b'"


class OAllele(Enum):
    XO = "X^O"
    Xo = "X^o"


class DAllele(Enum):
    D = "D"
    d = "d"


class AAllele(Enum):
    A = "A"
    a = "a"


class WAllele(Enum):
    W = "W"
    w = "w"


class Sex(Enum):
    MALE = "Male"
    FEMALE = "Female"


# Dominance rank: lower index = more dominant
B_RANK: dict[BAllele, int] = {BAllele.B: 0, BAllele.b: 1, BAllele.bprime: 2}
O_RANK: dict[OAllele, int] = {OAllele.XO: 0, OAllele.Xo: 1}
D_RANK: dict[DAllele, int] = {DAllele.D: 0, DAllele.d: 1}
A_RANK: dict[AAllele, int] = {AAllele.A: 0, AAllele.a: 1}
W_RANK: dict[WAllele, int] = {WAllele.W: 0, WAllele.w: 1}


def canonical_pair(a1, a2, rank_map: dict) -> tuple:
    """Return (a1, a2) with the more-dominant allele first."""
    if rank_map[a1] <= rank_map[a2]:
        return (a1, a2)
    return (a2, a1)


@dataclass(frozen=True)
class Genotype:
    sex: Sex
    B_locus: Tuple[BAllele, BAllele]
    D_locus: Tuple[DAllele, DAllele]
    A_locus: Tuple[AAllele, AAllele]
    W_locus: Tuple[WAllele, WAllele]
    # len=1 for males (hemizygous), len=2 for females
    O_locus: tuple

    def __post_init__(self):
        if self.sex == Sex.MALE and len(self.O_locus) != 1:
            raise ValueError("Male must have exactly one O allele")
        if self.sex == Sex.FEMALE and len(self.O_locus) != 2:
            raise ValueError("Female must have exactly two O alleles")


# Population allele frequencies (Hardy-Weinberg priors)
# Approximate values; empirical sources: Lyons lab, UC Davis VGL, breed standards.
# Note: Siamese/Ragdoll colorpoint requires the C locus (not modelled here).
BREED_FREQS: dict[str, dict] = {
    "Domestic Shorthair": {
        "B": {BAllele.B: 0.80, BAllele.b: 0.15, BAllele.bprime: 0.05},
        "O": {OAllele.XO: 0.20, OAllele.Xo: 0.80},
        "D": {DAllele.D: 0.70, DAllele.d: 0.30},
        "A": {AAllele.A: 0.60, AAllele.a: 0.40},
        "W": {WAllele.W: 0.05, WAllele.w: 0.95},
    },
    "Maine Coon": {
        "B": {BAllele.B: 0.80, BAllele.b: 0.12, BAllele.bprime: 0.08},
        "O": {OAllele.XO: 0.30, OAllele.Xo: 0.70},
        "D": {DAllele.D: 0.75, DAllele.d: 0.25},
        "A": {AAllele.A: 0.75, AAllele.a: 0.25},
        "W": {WAllele.W: 0.04, WAllele.w: 0.96},
    },
    "Persian": {
        "B": {BAllele.B: 0.60, BAllele.b: 0.30, BAllele.bprime: 0.10},
        "O": {OAllele.XO: 0.25, OAllele.Xo: 0.75},
        "D": {DAllele.D: 0.50, DAllele.d: 0.50},
        "A": {AAllele.A: 0.35, AAllele.a: 0.65},
        "W": {WAllele.W: 0.12, WAllele.w: 0.88},
    },
    "Abyssinian": {
        # Strongly agouti (ticked tabby); more cinnamon/chocolate than average
        "B": {BAllele.B: 0.55, BAllele.b: 0.25, BAllele.bprime: 0.20},
        "O": {OAllele.XO: 0.20, OAllele.Xo: 0.80},
        "D": {DAllele.D: 0.60, DAllele.d: 0.40},
        "A": {AAllele.A: 0.95, AAllele.a: 0.05},
        "W": {WAllele.W: 0.01, WAllele.w: 0.99},
    },
    "Russian Blue": {
        # Breed standard requires blue (B_ + dd); d near-fixed
        "B": {BAllele.B: 0.90, BAllele.b: 0.08, BAllele.bprime: 0.02},
        "O": {OAllele.XO: 0.10, OAllele.Xo: 0.90},
        "D": {DAllele.D: 0.10, DAllele.d: 0.90},
        "A": {AAllele.A: 0.55, AAllele.a: 0.45},
        "W": {WAllele.W: 0.01, WAllele.w: 0.99},
    },
    "Turkish Angora": {
        # Elevated W frequency; many individuals are white
        "B": {BAllele.B: 0.82, BAllele.b: 0.13, BAllele.bprime: 0.05},
        "O": {OAllele.XO: 0.25, OAllele.Xo: 0.75},
        "D": {DAllele.D: 0.60, DAllele.d: 0.40},
        "A": {AAllele.A: 0.55, AAllele.a: 0.45},
        "W": {WAllele.W: 0.45, WAllele.w: 0.55},
    },
}

# Derive "Unknown" as a uniform mixture over all named breeds.
# Used when breed identity is itself unobserved — the right prior is the
# average over the known breed distribution rather than any single breed.
def _avg_freqs(base: dict[str, dict]) -> dict:
    n = len(base)
    loci = ["B", "O", "D", "A", "W"]
    return {
        locus: {
            allele: sum(base[b][locus][allele] for b in base) / n
            for allele in next(iter(base.values()))[locus]
        }
        for locus in loci
    }

BREED_FREQS = {"Unknown": _avg_freqs(BREED_FREQS), **BREED_FREQS}
BREEDS: list[str] = list(BREED_FREQS.keys())  # "Unknown" is first

for _breed, _loci in BREED_FREQS.items():
    for _lname, _freq in _loci.items():
        assert abs(sum(_freq.values()) - 1.0) < 1e-9, f"{_breed} {_lname} freqs don't sum to 1"


def _hw_auto(a1, a2, freq: dict) -> float:
    if a1 == a2:
        return freq[a1] ** 2
    return 2.0 * freq[a1] * freq[a2]


def genotype_prior_for_breed(g: Genotype, breed: str) -> float:
    """Joint prior P(genotype) using breed-specific allele frequencies."""
    f = BREED_FREQS[breed]
    p = _hw_auto(g.B_locus[0], g.B_locus[1], f["B"])
    p *= _hw_auto(g.D_locus[0], g.D_locus[1], f["D"])
    p *= _hw_auto(g.A_locus[0], g.A_locus[1], f["A"])
    p *= _hw_auto(g.W_locus[0], g.W_locus[1], f["W"])
    if g.sex == Sex.FEMALE:
        p *= _hw_auto(g.O_locus[0], g.O_locus[1], f["O"])
    else:
        p *= f["O"][g.O_locus[0]]
    return p


def genotype_prior(g: Genotype) -> float:
    """Joint prior using Domestic Shorthair frequencies (default breed)."""
    return genotype_prior_for_breed(g, "Domestic Shorthair")


def _autosomal_pairs(allele_enum) -> list[tuple]:
    return list(combinations_with_replacement(allele_enum, 2))


def enumerate_genotypes(sex: Sex) -> Iterator[Genotype]:
    """Yield every possible Genotype for the given sex (324 male, 486 female)."""
    b_pairs = _autosomal_pairs(BAllele)   # 6
    d_pairs = _autosomal_pairs(DAllele)   # 3
    a_pairs = _autosomal_pairs(AAllele)   # 3
    w_pairs = _autosomal_pairs(WAllele)   # 3

    if sex == Sex.MALE:
        o_options = [(o,) for o in OAllele]  # 2
    else:
        o_options = list(combinations_with_replacement(OAllele, 2))  # 3

    for B, D, A, W, O in iproduct(b_pairs, d_pairs, a_pairs, w_pairs, o_options):
        B_c = canonical_pair(B[0], B[1], B_RANK)
        D_c = canonical_pair(D[0], D[1], D_RANK)
        A_c = canonical_pair(A[0], A[1], A_RANK)
        W_c = canonical_pair(W[0], W[1], W_RANK)
        O_c = canonical_pair(O[0], O[1], O_RANK) if sex == Sex.FEMALE else O
        yield Genotype(sex=sex, B_locus=B_c, D_locus=D_c, A_locus=A_c, W_locus=W_c, O_locus=O_c)
