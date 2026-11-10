import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq

st.set_page_config(
    page_title="AquaEquil Simulator",
    layout="wide"
)


GROUPS = {
    "OCl": {
        "species": ["HOCl", "OCl⁻"],
        "pKas": [7.53],
        "charges": [0, -1]
    },
    "CO₃": {
        "species": ["H₂CO₃", "HCO₃⁻", "CO₃²⁻"],
        "pKas": [6.35, 10.33],
        "charges": [0, -1, -2]
    },
    "SO₄": {
        "species": ["H₂SO₄", "HSO₄⁻", "SO₄²⁻"],
        "pKas": [-3.00, 1.99],
        "charges": [0, -1, -2]
    },
    "PO₄": {
        "species": ["H₃PO₄", "H₂PO₄⁻", "HPO₄²⁻", "PO₄³⁻"],
        "pKas": [2.15, 7.20, 12.35],
        "charges": [0, -1, -2, -3]
    }
}


REAGENTS = {
    "NaOH": {"groups": [], "positive": 1, "negative": 1},
    "KOH": {"groups": [], "positive": 1, "negative": 1},
    "LiOH": {"groups": [], "positive": 1, "negative": 1},
    "Ca(OH)₂": {"groups": [], "positive": 2, "negative": 2},
    "HCl": {"groups": [], "positive": 0, "negative": 1},
    "HNO₃": {"groups": [], "positive": 0, "negative": 1},
    "H₂SO₄": {"groups": ["SO₄"], "positive": 0, "negative": 2},
    "NaOCl": {"groups": ["OCl"], "positive": 1, "negative": 1},
    "KOCl": {"groups": ["OCl"], "positive": 1, "negative": 1},
    "Na₂CO₃": {"groups": ["CO₃"], "positive": 2, "negative": 2},
    "K₂CO₃": {"groups": ["CO₃"], "positive": 2, "negative": 2},
    "NaHCO₃": {"groups": ["CO₃"], "positive": 1, "negative": 1},
    "Na₂SO₄": {"groups": ["SO₄"], "positive": 2, "negative": 2},
    "K₂SO₄": {"groups": ["SO₄"], "positive": 2, "negative": 2},
    "Na₃PO₄": {"groups": ["PO₄"], "positive": 3, "negative": 3}
}


def distribution(h, total, pKas):
    ka_values = 10 ** (-np.asarray(pKas, dtype=float))
    n = len(ka_values)

    terms = [h ** n]

    for i in range(1, n + 1):
        terms.append(
            np.prod(ka_values[:i]) * h ** (n - i)
        )

    terms = np.asarray(terms)

    return total * terms / terms.sum()


def solve_general(
    selected_reagents,
    reagent_amounts,
    selected_groups,
    group_inputs,
    hcl_mM=0.0,
    kw=1e-14
):
    fixed_positive = 0.0
    fixed_negative = hcl_mM / 1000.0
    families = []

    for reagent in selected_reagents:
        amount = reagent_amounts[reagent]
        data = REAGENTS[reagent]

        fixed_positive += amount * data["positive"]
        fixed_negative += amount * data["negative"]

    for group in selected_groups:
        data = GROUPS[group]
        settings = group_inputs[group]

        total = settings["total"]
        pKas = settings["pKas"]

        families.append({
            "species": data["species"],
            "charges": data["charges"],
            "total": total,
            "pKas": pKas
        })

        fixed_positive += total * abs(min(data["charges"]))

    def charge_balance(log_h):
        h = 10 ** log_h
        oh = kw / h

        positive = h + fixed_positive
        negative = oh + fixed_negative

        for family in families:
            amounts = distribution(
                h,
                family["total"],
                family["pKas"]
            )

            for amount, charge in zip(
                amounts,
                family["charges"]
            ):
                if charge < 0:
                    negative += amount * abs(charge)

        return positive - negative

    log_h = brentq(charge_balance, -14, 0)

    h = 10 ** log_h
    oh = kw / h

    species = {
        "H⁺": h,
        "OH⁻": oh
    }

    for family in families:
        amounts = distribution(
            h,
            family["total"],
            family["pKas"]
        )

        for name, amount in zip(
            family["species"],
            amounts
        ):
            species[name] = amount

    return {
        "pH": -np.log10(h),
        "species": species,
        "charge_residual": charge_balance(log_h)
    }


def original_equilibrium(
    naoh,
    naocl,
    hcl_mM,
    pka,
    kw
):
    hcl = hcl_mM / 1000
    ka = 10 ** (-pka)

    sodium = naoh + naocl
    chlorine_total = naocl
    chloride = hcl

    def charge_balance(log_h):
        h = 10 ** log_h
        oh = kw / h
        ocl = chlorine_total * ka / (ka + h)

        return h + sodium - oh - ocl - chloride

    log_h = brentq(charge_balance, -14, 0)

    h = 10 ** log_h
    oh = kw / h
    ocl = chlorine_total * ka / (ka + h)
    hocl = chlorine_total - ocl

    return {
        "pH": -np.log10(h),
        "H": h,
        "OH": oh,
        "HOCl": hocl,
        "OCl": ocl,
        "charge_residual": charge_balance(log_h),
        "mass_error": hocl + ocl - chlorine_total
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Reagents")

selected_reagents = st.sidebar.multiselect(
    "Select reagent(s)",
    list(REAGENTS.keys()),
    default=[]
)

reagent_amounts = {}

for reagent in selected_reagents:
    reagent_amounts[reagent] = st.sidebar.number_input(
        f"{reagent} amount (M)",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key=f"amount_{reagent}"
    )


if len(selected_reagents) == 1:
    related_groups = set(
        REAGENTS[selected_reagents[0]]["groups"]
    )

    if related_groups:
        visible_groups = [
            group
            for group in GROUPS
            if group in related_groups
        ]
    else:
        visible_groups = list(GROUPS.keys())
else:
    visible_groups = list(GROUPS.keys())


st.sidebar.header("Species Groups")

selected_groups = st.sidebar.multiselect(
    "Select species group(s)",
    visible_groups,
    default=[]
)

group_inputs = {}

for group in selected_groups:
    st.sidebar.markdown(f"### {group}")

    total = st.sidebar.number_input(
        f"{group} total concentration (M)",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key=f"total_{group}"
    )

    pKas = GROUPS[group]["pKas"]

    st.sidebar.markdown("Assigned pKa values")

    for index, pka in enumerate(pKas):
        st.sidebar.number_input(
            f"{group} pKa {index + 1}",
            value=float(pka),
            step=0.1,
            format="%.2f",
            disabled=True,
            key=f"display_pka_{group}_{index}"
        )

    group_inputs[group] = {
        "total": total,
        "pKas": pKas
    }


inputs_ready = (
    bool(selected_reagents)
    and bool(selected_groups)
    and all(
        amount is not None
        for amount in reagent_amounts.values()
    )
    and all(
        settings["total"] is not None
        for settings in group_inputs.values()
    )
)


main_page, validation_page = st.tabs(
    ["Main", "Validation"]
)


# ============================================================
# MAIN PAGE
# ============================================================

with main_page:
    st.title("Main")

    if not inputs_ready:
        st.info(
            "Choose reagent(s) and species group(s), "
            "then enter all concentrations."
        )
    else:
        result = solve_general(
            selected_reagents,
            reagent_amounts,
            selected_groups,
            group_inputs,
            hcl_mM=0.0,
            kw=1e-14
        )

        st.subheader("Equilibrium Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Equilibrium pH",
                f"{result['pH']:.5f}"
            )

        with col2:
            dominant = max(
                result["species"],
                key=result["species"].get
            )

            st.metric(
                "Largest Species",
                dominant
            )

        result_table = pd.DataFrame({
            "Species": list(result["species"].keys()),
            "Concentration (M)": list(
                result["species"].values()
            )
        })

        st.dataframe(
            result_table.style.format({
                "Concentration (M)": "{:.5f}"
            }),
            use_container_width=True
        )

        st.write(
            "Charge residual:",
            f"{result['charge_residual']:.6e} M"
        )

        st.header("Visualization")

        hcl_values = np.linspace(
            0.0,
            1.9,
            120
        )

        curve = [
            solve_general(
                selected_reagents,
                reagent_amounts,
                selected_groups,
                group_inputs,
                hcl_mM=hcl_value,
                kw=1e-14
            )
            for hcl_value in hcl_values
        ]

        st.subheader("pH versus HCl concentration")

        fig_pH, ax_pH = plt.subplots(
            figsize=(10, 5)
        )

        ax_pH.plot(
            hcl_values,
            [
                item["pH"]
                for item in curve
            ],
            linewidth=2,
            label="Calculated pH"
        )

        if (
            "NaOH" in selected_reagents
            and "NaOCl" in selected_reagents
        ):
            validation_x = [
                0.00,
                0.50,
                1.00,
                1.50,
                1.90
            ]

            validation_y = [
                original_equilibrium(
                    reagent_amounts["NaOH"],
                    reagent_amounts["NaOCl"],
                    hcl,
                    7.53,
                    1e-14
                )["pH"]
                for hcl in validation_x
            ]

            ax_pH.scatter(
                validation_x,
                validation_y,
                color="red",
                marker="x",
                s=70,
                linewidths=2,
                label="Validation points"
            )

        ax_pH.set_title(
            "pH versus HCl concentration"
        )
        ax_pH.set_xlabel(
            "HCl concentration (mM)"
        )
        ax_pH.set_ylabel("pH")
        ax_pH.set_ylim(0, 14)
        ax_pH.grid(alpha=0.3)

        ax_pH.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

        fig_pH.tight_layout()
        st.pyplot(fig_pH)

        st.subheader(
            "Equilibrium species concentration versus HCl concentration"
        )

        fig_species, ax_species = plt.subplots(
            figsize=(10, 6)
        )

        for species_name in result["species"]:
            ax_species.plot(
                hcl_values,
                [
                    item["species"].get(
                        species_name,
                        0.0
                    ) * 1000
                    for item in curve
                ],
                linewidth=2,
                label=species_name
            )

        ax_species.set_title(
            "Equilibrium species concentration versus HCl concentration"
        )
        ax_species.set_xlabel(
            "HCl concentration (mM)"
        )
        ax_species.set_ylabel(
            "Concentration (mM)"
        )
        ax_species.grid(alpha=0.3)

        ax_species.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False
        )

        fig_species.tight_layout()
        st.pyplot(fig_species)

        graph_buffer = io.BytesIO()

        fig_species.savefig(
            graph_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        st.download_button(
            "Download Visualization Graphs as PNG",
            graph_buffer.getvalue(),
            "AquaEquil_visualization.png",
            "image/png"
        )


# ============================================================
# VALIDATION PAGE
# ============================================================

with validation_page:
    st.title("Validation")

    st.info(
        "Enter the original validation values "
        "and press Run Validation."
    )

    validation_naoh = st.number_input(
        "NaOH concentration (M)",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key="validation_naoh"
    )

    validation_naocl = st.number_input(
        "NaOCl concentration (M)",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key="validation_naocl"
    )

    validation_pka = st.number_input(
        "HOCl pKa",
        min_value=-20.0,
        max_value=20.0,
        value=None,
        step=0.1,
        format="%.2f",
        key="validation_pka"
    )

    validation_kw = st.number_input(
        "Kw",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key="validation_kw"
    )

    run_validation = st.button(
        "Run Validation",
        type="primary"
    )

    if not run_validation:
        st.info(
            "No validation values or graphs are shown "
            "until Run Validation is pressed."
        )

    elif any(
        value is None
        for value in [
            validation_naoh,
            validation_naocl,
            validation_pka,
            validation_kw
        ]
    ):
        st.warning(
            "Enter all validation values first."
        )

    else:
        validation_cases = [
            0.00,
            0.50,
            1.00,
            1.50,
            1.90
        ]

        expected_pH = {
            0.00: 11.00,
            0.50: 10.70,
            1.00: 9.26,
            1.50: 7.53,
            1.90: 6.58
        }

        rows = []

        for hcl in validation_cases:
            current = original_equilibrium(
                validation_naoh,
                validation_naocl,
                hcl,
                validation_pka,
                validation_kw
            )

            difference = (
                current["pH"]
                - expected_pH[hcl]
            )

            passed = (
                abs(difference) <= 0.10
                and abs(
                    current["charge_residual"]
                ) < 1e-7
                and abs(
                    current["mass_error"]
                ) < 1e-7
            )

            rows.append({
                "HCl (mM)": hcl,
                "Calculated pH": current["pH"],
                "Expected pH": expected_pH[hcl],
                "Difference": difference,
                "Charge residual (M)": current[
                    "charge_residual"
                ],
                "Mass error (M)": current[
                    "mass_error"
                ],
                "Status": (
                    "PASS"
                    if passed
                    else "CHECK"
                )
            })

        validation_df = pd.DataFrame(rows)

        st.dataframe(
            validation_df.style.format({
                "HCl (mM)": "{:.2f}",
                "Calculated pH": "{:.4f}",
                "Expected pH": "{:.2f}",
                "Difference": "{:+.4f}",
                "Charge residual (M)": "{:.2e}",
                "Mass error (M)": "{:.2e}"
            }),
            use_container_width=True
        )

        validation_x = np.linspace(
            0.0,
            1.9,
            120
        )

        validation_curve = [
            original_equilibrium(
                validation_naoh,
                validation_naocl,
                hcl,
                validation_pka,
                validation_kw
            )
            for hcl in validation_x
        ]

        fig_validation, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8)
        )

        axes[0].plot(
            validation_x,
            [
                item["pH"]
                for item in validation_curve
            ],
            linewidth=2,
            label="Calculated pH"
        )

        axes[0].scatter(
            validation_cases,
            [
                original_equilibrium(
                    validation_naoh,
                    validation_naocl,
                    hcl,
                    validation_pka,
                    validation_kw
                )["pH"]
                for hcl in validation_cases
            ],
            color="red",
            marker="x",
            s=70,
            label="Validation points"
        )

        axes[0].set_title(
            "Original Validation Titration Curve"
        )
        axes[0].set_xlabel(
            "HCl addition (mM)"
        )
        axes[0].set_ylabel("pH")
        axes[0].set_ylim(0, 14)
        axes[0].grid(alpha=0.3)

        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

        axes[1].plot(
            validation_x,
            [
                item["HOCl"] * 1000
                for item in validation_curve
            ],
            linewidth=2,
            label="HOCl"
        )

        axes[1].plot(
            validation_x,
            [
                item["OCl"] * 1000
                for item in validation_curve
            ],
            linewidth=2,
            label="OCl⁻"
        )

        axes[1].set_title(
            "Original HOCl/OCl⁻ Speciation"
        )
        axes[1].set_xlabel(
            "HCl addition (mM)"
        )
        axes[1].set_ylabel(
            "Concentration (mM)"
        )
        axes[1].grid(alpha=0.3)

        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

        fig_validation.tight_layout()
        st.pyplot(fig_validation)

        validation_buffer = io.BytesIO()

        fig_validation.savefig(
            validation_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        st.download_button(
            "Download Validation Graphs as PNG",
            validation_buffer.getvalue(),
            "AquaEquil_validation_graphs.png",
            "image/png"
        )
