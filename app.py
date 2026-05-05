from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from breeding import predict_offspring
from genetics import BREEDS, Sex
from inference import infer_genotype_dist
from phenotype import FEMALE_PHENOTYPES, MALE_PHENOTYPES
from reverse import marginalise_parents, reverse_infer

st.set_page_config(
    page_title="Feline Genotype Inference Engine",
    page_icon="🐱",
    layout="wide",
)

st.title("Feline Genotype Inference Engine")
st.caption(
    "Bayesian inference over hidden genotypes, applied to Mendelian cat genetics. "
    "Observe coat color → infer genotype distribution → predict offspring (or parents)."
)

tab_forward, tab_reverse = st.tabs(["Predict Offspring →", "← Identify Parents"])

# ── helpers ──────────────────────────────────────────────────────────────────

def _prob_table(probs: dict[str, float]) -> pd.DataFrame:
    df = (
        pd.DataFrame.from_dict(probs, orient="index", columns=["Probability"])
        .sort_values("Probability", ascending=False)
    )
    df.index.name = "Coat Color"
    return df


def _pie_chart(probs: dict[str, float], title: str = "") -> None:
    df = pd.DataFrame({"Coat Color": list(probs.keys()), "Probability": list(probs.values())})
    df = df.sort_values("Probability", ascending=False)
    fig = px.pie(
        df,
        names="Coat Color",
        values="Probability",
        title=title,
        hole=0.35,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=48, b=8, l=8, r=8),
        title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_offspring_col(
    col: st.delta_generator.DeltaGenerator,
    label: str,
    probs: dict[str, float],
) -> None:
    with col:
        st.subheader(f"{label} offspring")
        if not probs:
            st.info("No offspring of this sex possible.")
            return
        _pie_chart(probs)
        df_display = _prob_table(probs).copy()
        df_display["Probability"] = df_display["Probability"].map("{:.1%}".format)
        st.dataframe(df_display, use_container_width=True)


# ── Tab 1: Forward inference ──────────────────────────────────────────────────

with tab_forward:
    st.markdown("Select two parent coat colors (and optionally their breeds) to predict offspring color probabilities.")

    col_sire, col_dam = st.columns(2)

    _dsh_idx = BREEDS.index("Domestic Shorthair")

    with col_sire:
        st.subheader("Sire (Father)")
        sire_breed = st.selectbox("Breed", BREEDS, index=_dsh_idx, key="sire_breed")
        sire_phenotype = st.selectbox("Coat color", MALE_PHENOTYPES, key="sire_ph")

    with col_dam:
        st.subheader("Dam (Mother)")
        dam_breed = st.selectbox("Breed", BREEDS, index=_dsh_idx, key="dam_breed")
        dam_phenotype = st.selectbox("Coat color", FEMALE_PHENOTYPES, key="dam_ph")

    if st.button("Compute Offspring Distribution", type="primary", key="fwd_btn"):
        with st.spinner("Running Bayesian inference and Mendelian cross..."):
            sire_dist = infer_genotype_dist(sire_phenotype, Sex.MALE, sire_breed)
            dam_dist = infer_genotype_dist(dam_phenotype, Sex.FEMALE, dam_breed)
            male_probs, female_probs = predict_offspring(sire_dist, dam_dist)

        st.markdown("---")
        st.subheader("Offspring Color Distribution")
        st.caption("Offspring sex ratio is always 50/50. Probabilities below are conditional on sex.")

        res_male, res_female = st.columns(2)
        _render_offspring_col(res_male, "Male", male_probs)
        _render_offspring_col(res_female, "Female", female_probs)

        with st.expander("How this works"):
            st.markdown(
                """
**Loci modelled:** B (black/chocolate/cinnamon), O (orange, X-linked), D (dilute),
A (agouti/tabby), W (dominant white).

**Bayesian inference:** Coat color is observable but genotype is hidden.
The engine starts with Hardy–Weinberg priors for the selected breed and restricts
to genotypes compatible with the observed phenotype, then renormalises.
Breed priors are approximate values based on published genetics literature.

**Mendelian cross:** Offspring probabilities are computed by averaging over all
parent genotype pairs weighted by their posterior probabilities, with X-linkage
handled explicitly for the O locus.

**Why orange cats are usually male:** The O gene sits on the X chromosome.
A female needs two copies (X^O X^O) to be fully orange; one copy (X^O X^o) gives
tortoiseshell. Males only have one X, so a single X^O makes them fully orange.
"""
            )

# ── Tab 2: Reverse inference ──────────────────────────────────────────────────

with tab_reverse:
    st.markdown(
        "Given an offspring's coat color, infer what the parents probably looked like. "
        "Fix one parent to narrow the distribution for the other."
    )

    col_off, col_sex = st.columns(2)
    offspring_sex = col_sex.selectbox(
        "Offspring sex",
        [Sex.MALE, Sex.FEMALE],
        format_func=lambda s: s.value,
        key="rev_sex",
    )
    off_phenotypes = MALE_PHENOTYPES if offspring_sex == Sex.MALE else FEMALE_PHENOTYPES
    offspring_ph = col_off.selectbox("Offspring coat color", off_phenotypes, key="rev_ph")

    st.markdown("**Known parent (leave as Unknown to infer both)**")
    col_ks, col_kd = st.columns(2)
    known_sire_sel = col_ks.selectbox(
        "Sire coat color", ["Unknown"] + MALE_PHENOTYPES, key="known_sire"
    )
    known_dam_sel = col_kd.selectbox(
        "Dam coat color", ["Unknown"] + FEMALE_PHENOTYPES, key="known_dam"
    )

    col_sb, col_db = st.columns(2)
    sire_breed_default = 0 if known_sire_sel == "Unknown" else BREEDS.index("Domestic Shorthair")
    dam_breed_default  = 0 if known_dam_sel  == "Unknown" else BREEDS.index("Domestic Shorthair")
    rev_sire_breed = col_sb.selectbox("Sire breed prior", BREEDS, index=sire_breed_default, key="rev_sire_breed")
    rev_dam_breed  = col_db.selectbox("Dam breed prior",  BREEDS, index=dam_breed_default,  key="rev_dam_breed")

    known_sire_ph = None if known_sire_sel == "Unknown" else known_sire_sel
    known_dam_ph = None if known_dam_sel == "Unknown" else known_dam_sel

    if known_sire_ph and known_dam_ph:
        st.warning("At least one parent must be Unknown for reverse inference.")
    elif st.button("Identify Parents", type="primary", key="rev_btn"):
        both_unknown = not known_sire_ph and not known_dam_ph
        n_calls = (
            len(MALE_PHENOTYPES) * len(FEMALE_PHENOTYPES) if both_unknown
            else len(FEMALE_PHENOTYPES) if known_sire_ph
            else len(MALE_PHENOTYPES)
        )
        with st.spinner(f"Running reverse inference ({n_calls} forward passes)..."):
            result = reverse_infer(
                offspring_ph,
                offspring_sex,
                known_sire_ph=known_sire_ph,
                known_dam_ph=known_dam_ph,
                sire_breed=rev_sire_breed,
                dam_breed=rev_dam_breed,
            )

        st.markdown("---")

        if not result:
            st.error("No parent combination can produce this offspring phenotype. Check your inputs.")

        elif both_unknown:
            # Show marginals + top joint combinations
            sire_marg, dam_marg = marginalise_parents(result)

            st.subheader("Marginal parent distributions")
            mcol, dcol = st.columns(2)
            with mcol:
                st.markdown("**Most likely sire coat color**")
                _pie_chart(sire_marg)
                df_s = _prob_table(sire_marg)
                df_s["Probability"] = df_s["Probability"].map("{:.1%}".format)
                st.dataframe(df_s, use_container_width=True)
            with dcol:
                st.markdown("**Most likely dam coat color**")
                _pie_chart(dam_marg)
                df_d = _prob_table(dam_marg)
                df_d["Probability"] = df_d["Probability"].map("{:.1%}".format)
                st.dataframe(df_d, use_container_width=True)

            st.subheader("Top parent combinations (joint)")
            top = sorted(result.items(), key=lambda x: -x[1])[:12]
            df_joint = pd.DataFrame(
                [(f"{s} × {d}", f"{p:.1%}") for (s, d), p in top],
                columns=["Sire × Dam", "Probability"],
            )
            st.dataframe(df_joint, use_container_width=True, hide_index=True)

        else:
            unknown_label = "Dam" if known_sire_ph else "Sire"
            st.subheader(f"Most likely {unknown_label} coat color")
            _pie_chart(result)
            df = _prob_table(result)
            df["Probability"] = df["Probability"].map("{:.1%}".format)
            st.dataframe(df, use_container_width=True)

        with st.expander("How reverse inference works"):
            st.markdown("**Bayes' theorem applied to parents:**")
            st.latex(
                r"P(\text{parents} \mid \text{offspring}) \;\propto\; P(\text{offspring} \mid \text{parents}) \times P(\text{parents})"
            )
            st.markdown(
                "- $P(\\text{offspring} \\mid \\text{parents})$ — the forward model (Mendelian cross over inferred parent genotypes)\n"
                "- $P(\\text{parents})$ — the phenotype marginal under the breed's Hardy–Weinberg prior\n\n"
                "The prior $P(\\text{parents})$ ensures that rare parent phenotypes (cinnamon, fawn) get\n"
                "discounted even when they're genetically compatible with the offspring.\n"
                "With one parent fixed, the posterior collapses to a distribution over only the unknown parent."
            )
