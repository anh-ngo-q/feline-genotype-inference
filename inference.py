from __future__ import annotations

from functools import lru_cache

from genetics import (
    BREEDS,
    Genotype,
    Sex,
    enumerate_genotypes,
    genotype_prior_for_breed,
)
from phenotype import genotype_to_phenotype


@lru_cache(maxsize=None)
def infer_genotype_dist(
    phenotype: str,
    sex: Sex,
    breed: str = "Domestic Shorthair",
) -> dict[Genotype, float]:
    """Bayesian posterior P(genotype | phenotype, sex, breed).

    Likelihood is hard 0/1: a genotype either produces the phenotype or it
    doesn't. Posterior = prior restricted to compatible genotypes, renormalized.
    Result is cached — do not mutate the returned dict.
    """
    if breed not in BREEDS:
        raise ValueError(f"Unknown breed {breed!r}. Choose from: {BREEDS}")
    raw = {
        g: genotype_prior_for_breed(g, breed)
        for g in enumerate_genotypes(sex)
        if genotype_to_phenotype(g) == phenotype
    }
    if not raw:
        raise ValueError(
            f"Phenotype {phenotype!r} is impossible for sex {sex.value!r}. "
            f"Check MALE_PHENOTYPES / FEMALE_PHENOTYPES for valid inputs."
        )
    total = sum(raw.values())
    return {g: p / total for g, p in raw.items()}
