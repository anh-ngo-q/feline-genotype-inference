# Feline Genotype Inference Engine

**[Live demo → feline-genotype-inference.streamlit.app](https://feline-genotype-inference.streamlit.app/)**

A Streamlit app that predicts offspring coat color probabilities given two parent cat phenotypes — and can also infer what parents likely looked like from an offspring's coat color.

The core problem: you can observe a cat's coat color, but not its genotype. Genotype is what actually gets inherited. This app treats parent genotypes as latent variables, uses Bayesian inference to recover a distribution over them from the observed coat color, then applies Mendelian rules to predict offspring.

---

## Why this is interesting

Orange cats are almost always male. The reason: the orange gene (`O`) sits on the X chromosome. Males (XY) only carry one copy — one `X^O` allele makes them fully orange. Females (XX) need *two* copies to be orange; one `X^O` + one `X^o` produces tortoiseshell instead, because different cells silence different X chromosomes (X-inactivation), creating a mosaic of orange and non-orange patches.

That rabbit hole turns into a latent variable inference problem: you see the coat, not the genes. Modeling that gap explicitly is what this engine does.

---

## What it does

**Forward inference — predict offspring**
Select two parent coat colors (and optionally their breeds). The engine:
1. Infers a probability distribution over each parent's hidden genotype using Bayesian filtering with Hardy–Weinberg population priors
2. Applies Mendelian segregation (including X-linkage for the orange locus) to compute the offspring genotype distribution
3. Maps offspring genotypes to phenotypes and displays the probability breakdown

**Reverse inference — identify parents**
Select an offspring coat color. The engine inverts the breeding model to infer what the parents probably looked like — with one parent optionally fixed to narrow the distribution for the other.

---

## Genetics modelled

| Locus | Type | Effect |
|---|---|---|
| **B** | Autosomal | B > b > b′. Black / chocolate / cinnamon base pigment |
| **O** | **X-linked** | X^O > X^o. Orange epistatic to B; tortoiseshell in X^O X^o females |
| **D** | Autosomal | D > d. Dilute: black → blue, orange → cream, chocolate → lilac |
| **A** | Autosomal | A > a. Agouti (tabby pattern) vs. solid |
| **W** | Autosomal | W > w. Dominant white — masks all other loci |

Breed-specific allele frequency priors are available for Domestic Shorthair, Maine Coon, Persian, Abyssinian, Russian Blue, and Turkish Angora. An "Unknown" breed option uses a uniform mixture over all known breeds.

---

## Technical approach

```
Observed phenotype
      │
      ▼  Bayesian filtering
      │  P(genotype | phenotype) ∝ P(genotype) × P(phenotype | genotype)
      │  Likelihood: hard 0/1 (deterministic phenotype map)
      │  Prior: Hardy–Weinberg allele frequencies per breed
      ▼
Genotype posterior (sire) × Genotype posterior (dam)
      │
      ▼  Mendelian cross + marginalization
      │  P(offspring) = ΣΣ P(sire_g) × P(dam_g) × P(offspring | sire_g, dam_g)
      │  O locus handled with explicit X-linkage and sex determination
      ▼
Offspring phenotype distribution
```

Reverse inference applies Bayes' theorem to the forward model:

```
P(parents | offspring) ∝ P(offspring | parents) × P(parents)
```

where `P(offspring | parents)` is the forward model and `P(parents)` is the phenotype marginal under the breed prior.

---

## Project structure

```
├── genetics.py      # Allele enums, Genotype dataclass, HW priors, breed frequencies
├── phenotype.py     # genotype → phenotype mapping, valid phenotype enumeration
├── inference.py     # Bayesian posterior P(genotype | phenotype, sex, breed)
├── breeding.py      # Mendelian cross, predict_offspring
├── reverse.py       # Reverse inference: P(parents | offspring)
├── app.py           # Streamlit UI
└── requirements.txt
```

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+.

---

## Model assumptions and limitations

- Hardy–Weinberg equilibrium (random mating, large population) — violated by selective breeding
- Loci are independent — physical linkage exists but is small for these loci
- Phenotype observation is noiseless — the model assumes you know the coat color exactly
- Five loci only — colorpoint (C locus, Siamese pattern), white spotting (S locus), and tabby pattern type (T locus) are not modelled

Breed priors are approximate values based on published genetics literature and breed standards, not empirically measured allele frequencies.
