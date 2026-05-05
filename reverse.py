from __future__ import annotations

from functools import lru_cache

from genetics import BREEDS, Sex, enumerate_genotypes, genotype_prior_for_breed
from phenotype import FEMALE_PHENOTYPES, MALE_PHENOTYPES, genotype_to_phenotype
from inference import infer_genotype_dist
from breeding import predict_offspring


@lru_cache(maxsize=None)
def _forward_probs(
    sire_ph: str,
    dam_ph: str,
    sire_breed: str = "Domestic Shorthair",
    dam_breed: str = "Domestic Shorthair",
) -> tuple[dict[str, float], dict[str, float]]:
    """Cached forward prediction for a (sire_phenotype, dam_phenotype) pair.

    Returns (male_offspring_probs, female_offspring_probs), each summing to 1.
    Keyed by phenotype strings + breeds so the dict-hashability issue is avoided.
    """
    sire_dist = infer_genotype_dist(sire_ph, Sex.MALE, sire_breed)
    dam_dist = infer_genotype_dist(dam_ph, Sex.FEMALE, dam_breed)
    return predict_offspring(sire_dist, dam_dist)


@lru_cache(maxsize=None)
def phenotype_prior_weight(
    phenotype: str,
    sex: Sex,
    breed: str = "Domestic Shorthair",
) -> float:
    """P(phenotype) under the HW prior for the given breed.

    This is the marginal probability of observing this coat color in a random
    cat of the given sex from the given breed population.
    """
    return sum(
        genotype_prior_for_breed(g, breed)
        for g in enumerate_genotypes(sex)
        if genotype_to_phenotype(g) == phenotype
    )


def reverse_infer(
    offspring_ph: str,
    offspring_sex: Sex,
    known_sire_ph: str | None = None,
    known_dam_ph: str | None = None,
    sire_breed: str = "Domestic Shorthair",
    dam_breed: str = "Domestic Shorthair",
) -> dict:
    """Infer parent phenotype distribution given an observed offspring phenotype.

    Uses Bayes' theorem:
        P(parents | offspring) ∝ P(offspring | parents) × P(parents)

    where P(offspring | parents) comes from the forward model and P(parents)
    is the phenotype marginal under the breed's HW prior.

    Returns:
        Both parents unknown  → {(sire_ph, dam_ph): probability}
        Sire known, dam unknown → {dam_ph: probability}
        Dam known, sire unknown → {sire_ph: probability}
    """
    unnorm: dict = {}

    if known_sire_ph and known_dam_ph:
        raise ValueError("At least one parent must be unknown.")

    if known_sire_ph:
        # Sire known — infer dam phenotype distribution
        for dam_ph in FEMALE_PHENOTYPES:
            male_p, female_p = _forward_probs(known_sire_ph, dam_ph, sire_breed, dam_breed)
            probs = male_p if offspring_sex == Sex.MALE else female_p
            p_off = probs.get(offspring_ph, 0.0)
            if p_off > 0:
                unnorm[dam_ph] = phenotype_prior_weight(dam_ph, Sex.FEMALE, dam_breed) * p_off

    elif known_dam_ph:
        # Dam known — infer sire phenotype distribution
        for sire_ph in MALE_PHENOTYPES:
            male_p, female_p = _forward_probs(sire_ph, known_dam_ph, sire_breed, dam_breed)
            probs = male_p if offspring_sex == Sex.MALE else female_p
            p_off = probs.get(offspring_ph, 0.0)
            if p_off > 0:
                unnorm[sire_ph] = phenotype_prior_weight(sire_ph, Sex.MALE, sire_breed) * p_off

    else:
        # Both parents unknown — infer joint distribution over parent phenotype pairs
        for sire_ph in MALE_PHENOTYPES:
            p_sire = phenotype_prior_weight(sire_ph, Sex.MALE, sire_breed)
            for dam_ph in FEMALE_PHENOTYPES:
                male_p, female_p = _forward_probs(sire_ph, dam_ph, sire_breed, dam_breed)
                probs = male_p if offspring_sex == Sex.MALE else female_p
                p_off = probs.get(offspring_ph, 0.0)
                if p_off > 0:
                    p_dam = phenotype_prior_weight(dam_ph, Sex.FEMALE, dam_breed)
                    unnorm[(sire_ph, dam_ph)] = p_sire * p_dam * p_off

    total = sum(unnorm.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in unnorm.items()}


def marginalise_parents(
    joint: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[str, float]]:
    """From a joint (sire_ph, dam_ph) distribution, get marginals for each parent."""
    sire_marginal: dict[str, float] = {}
    dam_marginal: dict[str, float] = {}
    for (sire_ph, dam_ph), p in joint.items():
        sire_marginal[sire_ph] = sire_marginal.get(sire_ph, 0.0) + p
        dam_marginal[dam_ph] = dam_marginal.get(dam_ph, 0.0) + p
    return sire_marginal, dam_marginal
