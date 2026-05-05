from __future__ import annotations

from itertools import product as iproduct

from genetics import (
    A_RANK,
    B_RANK,
    D_RANK,
    O_RANK,
    W_RANK,
    Genotype,
    Sex,
    canonical_pair,
)
from phenotype import genotype_to_phenotype


def _cross_locus(p1_pair: tuple, p2_pair: tuple, rank_map: dict) -> dict[tuple, float]:
    """Mendelian cross for one autosomal diploid locus.

    Each of the 4 allele combinations gets prob 0.25; canonical pairs
    accumulate so the result always sums to 1.0.
    """
    result: dict[tuple, float] = {}
    for a in p1_pair:
        for b in p2_pair:
            key = canonical_pair(a, b, rank_map)
            result[key] = result.get(key, 0.0) + 0.25
    return result


def _cross_o_sex(sire: Genotype, dam: Genotype) -> dict[tuple, float]:
    """X-linkage + sex determination: returns {(Sex, o_tuple): prob}.

    Sire contributes Y (-> male) or his X (-> female), each 50%.
    Dam contributes either of her two O alleles, each 50%.
    Each of the 4 branches therefore has prob 0.25; total sums to 1.0.
    """
    result: dict[tuple, float] = {}
    o_sire_x = sire.O_locus[0]

    for dam_allele in dam.O_locus:
        # Y from sire -> male offspring carries only the dam's X allele
        male_key = (Sex.MALE, (dam_allele,))
        result[male_key] = result.get(male_key, 0.0) + 0.25

        # X from sire -> female offspring gets sire's X + dam's X
        o_female = canonical_pair(o_sire_x, dam_allele, O_RANK)
        female_key = (Sex.FEMALE, o_female)
        result[female_key] = result.get(female_key, 0.0) + 0.25

    return result


def cross_genotypes(sire: Genotype, dam: Genotype) -> dict[Genotype, float]:
    """Full Mendelian cross between two fixed parent genotypes.

    Returns a probability distribution over offspring Genotypes summing to 1.0.
    Assumes sire is Male and dam is Female.
    """
    b_dist = _cross_locus(sire.B_locus, dam.B_locus, B_RANK)
    d_dist = _cross_locus(sire.D_locus, dam.D_locus, D_RANK)
    a_dist = _cross_locus(sire.A_locus, dam.A_locus, A_RANK)
    w_dist = _cross_locus(sire.W_locus, dam.W_locus, W_RANK)
    sex_o_dist = _cross_o_sex(sire, dam)

    offspring: dict[Genotype, float] = {}
    for (b, p_b), (d, p_d), (a, p_a), (w, p_w), ((sex, o), p_o) in iproduct(
        b_dist.items(),
        d_dist.items(),
        a_dist.items(),
        w_dist.items(),
        sex_o_dist.items(),
    ):
        g = Genotype(sex=sex, B_locus=b, D_locus=d, A_locus=a, W_locus=w, O_locus=o)
        offspring[g] = offspring.get(g, 0.0) + p_b * p_d * p_a * p_w * p_o

    return offspring


def predict_offspring(
    sire_dist: dict[Genotype, float],
    dam_dist: dict[Genotype, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute offspring phenotype distributions, conditioned on sex.

    Marginalises over parent genotype uncertainty:
        P(phenotype | sex) ∝ ΣΣ P(sire_g) × P(dam_g) × P(offspring | sire_g, dam_g)

    Returns (male_probs, female_probs), each normalised to sum to 1.0.
    The unconditional sex split is always 50/50.
    """
    male_raw: dict[str, float] = {}
    female_raw: dict[str, float] = {}

    for sire_g, p_sire in sire_dist.items():
        for dam_g, p_dam in dam_dist.items():
            weight = p_sire * p_dam
            if weight == 0.0:
                continue
            for off_g, p_off in cross_genotypes(sire_g, dam_g).items():
                phenotype = genotype_to_phenotype(off_g)
                contribution = weight * p_off
                if off_g.sex == Sex.MALE:
                    male_raw[phenotype] = male_raw.get(phenotype, 0.0) + contribution
                else:
                    female_raw[phenotype] = female_raw.get(phenotype, 0.0) + contribution

    male_total = sum(male_raw.values())
    female_total = sum(female_raw.values())

    male_probs = {k: v / male_total for k, v in male_raw.items()} if male_total > 0 else {}
    female_probs = {k: v / female_total for k, v in female_raw.items()} if female_total > 0 else {}

    return male_probs, female_probs
