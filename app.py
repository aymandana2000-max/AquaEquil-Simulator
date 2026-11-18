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


# ============================================================
# SPECIES GROUPS
# ============================================================

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


# ============================================================
# REAGENTS
# ============================================================

REAGENTS = {
    "NaOH": {
        "positive": 1,
        "negative": 1,
        "groups": []
    },
    "KOH": {
        "positive": 1,
        "negative": 1,
        "groups": []
    },
    "LiOH": {
        "positive": 1,
        "negative": 1,
        "groups": []
    },
    "Ca(OH)₂": {
        "positive": 2,
        "negative": 2,
        "groups": []
    },
    "HCl": {
        "positive": 0,
        "negative": 1,
        "groups": []
    },
    "HNO₃": {
        "positive": 0,
        "negative": 1,
        "groups": []
    },
    "H₂SO₄": {
        "positive": 0,
        "negative": 2,
        "groups": ["SO₄"]
    },
    "NaOCl": {
        "positive": 1,
        "negative": 1,
        "groups": ["OCl"]
    },
    "KOCl": {
        "positive": 1,
        "negative": 1,
        "groups": ["OCl"]
    },
    "Na₂CO₃": {
        "positive": 2,
        "negative": 2,
        "groups": ["CO₃"]
    },
    "K₂CO₃": {
        "positive": 2,
        "negative": 2,
        "groups": ["CO₃"]
    },
    "NaHCO₃": {
        "positive": 1,
        "negative": 1,
        "groups": ["CO₃"]
    },
    "Na₂SO₄": {
        "positive": 2,
        "negative": 2,
        "groups": ["SO₄"]
    },
    "K₂SO₄": {
        "positive": 2,
        "negative": 2,
        "groups": ["SO₄"]
    },
    "Na₃PO₄": {
        "positive": 3,
        "negative": 3,
        "groups": ["PO₄"]
    }
}


# ============================================================
# CHEMISTRY FUNCTIONS
# ============================================================

def distribution(h, total, pKas):
    ka_values = 10 ** (-np.asarray(pKas, dtype=float))
    n = len(ka_values)

    terms = [h ** n]

    for i in range(1, n + 1):
        terms.append(
            np.prod(ka_values[:i])
            * h ** (n - i)
        )

    terms = np.asarray(terms)

    return total * terms / terms.sum()


def groups_from_reagents(selected_reagents, reagent_amounts):
    totals = {}

    for reagent in selected_reagents:
        amount = reagent_amounts[reagent]

        for group in REAGENTS[reagent]["groups"]:
            totals[group] = totals.get(group, 0.0) + amount

    return totals


def solve_general(
    selected_reagents,
    reagent_amounts,
    group_totals,
    extra_reagent=None,
    extra_amount=0.0,
    kw=1e-14
):
    amounts = dict(reagent_amounts)

    if extra_reagent is not None:
        amounts[extra_reagent] = extra_amount

    fixed_positive = 0.0
    fixed_negative = 0.0

    for reagent in selected_reagents:
        amount = amounts[reagent]
        data = REAGENTS[reagent]

        fixed_positive += amount * data["positive"]
        fixed_negative += amount * data["negative"]

    adjusted_totals = {}

    for reagent in selected_reagents:
        amount = amounts[reagent]

        for group in REAGENTS[reagent]["groups"]:
            adjusted_totals[group] = (
                adjusted_totals.get(group, 0.0)
                + amount
            )

    def charge_balance(log_h):
        h = 10 ** log_h
        oh = kw / h

        positive = h + fixed_positive
        negative = oh + fixed_negative

        for group, total in adjusted_totals.items():
            data = GROUPS[group]
            values = distribution(
                h,
                total,
                data["pKas"]
            )

            for value, charge in zip(
                values,
                data["charges"]
            ):
                if charge < 0:
                    negative += value * abs(charge)

        return positive - negative

    log_h = brentq(charge_balance, -14, 0)

    h = 10 ** log_h
    oh = kw / h

    species = {
        "H⁺": h,
        "OH⁻": oh
    }

    for group, total in adjusted_totals.items():
        data = GROUPS[group]
        values = distribution(
            h,
            total,
            data["pKas"]
        )

        for name, value in zip(
            data["species"],
            values
        ):
            species[name] = value

    return {
        "pH": -np.log10(h),
        "species": species,
        "charge_residual": charge_balance(log_h)
    }


def original_equilibrium(
    naoh,
    naocl,
    hcl_mM,
    pka=7.53,
    kw=1e-14
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


selected_amounts_ready = (
    bool(selected_reagents)
    and all(
        amount is not None
        for amount in reagent_amounts.values()
    )
)

group_totals = {}

if selected_amounts_ready:
    group_totals = groups_from_reagents(
        selected_reagents,
        reagent_amounts
    )

    if group_totals:
        st.sidebar.header("Automatically Included Species")

        for group, total in group_totals.items():
            st.sidebar.markdown(f"### {group}")

            st.sidebar.number_input(
                f"{group} total concentration (M)",
                value=float(total),
                format="%.5f",
                disabled=True,
                key=f"fixed_total_{group}"
            )

            st.sidebar.markdown("Assigned pKa values")

            for index, pka in enumerate(
                GROUPS[group]["pKas"]
            ):
                st.sidebar.number_input(
                    f"{group} pKa {index + 1}",
                    value=float(pka),
                    step=0.1,
                    format="%.2f",
                    disabled=True,
                    key=f"fixed_pka_{group}_{index}"
                )


# ============================================================
# PAGE TABS
# ============================================================

main_page, validation_page, about_page = st.tabs(
    [
        "Main",
        "Validation",
        "About"
    ]
)


# ============================================================
# MAIN PAGE
# ============================================================

with main_page:
    st.title("Main")

    if not selected_reagents:
        st.info(
            "Choose at least one reagent and enter its concentration."
        )

    elif not selected_amounts_ready:
        st.info(
            "Enter the concentration for every selected reagent."
        )

    else:
        result = solve_general(
            selected_reagents,
            reagent_amounts,
            group_totals
        )

        st.subheader("Equilibrium Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Equilibrium pH",
                f"{result['pH']:.5f}"
            )

        with col2:
            largest_species = max(
                result["species"],
                key=result["species"].get
            )

            st.metric(
                "Largest Species",
                largest_species
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

        # ----------------------------------------------------
        # Interaction graph: species versus pH
        # ----------------------------------------------------

        st.subheader(
            "Equilibrium species distribution versus pH"
        )

        pH_values = np.linspace(
            0,
            14,
            250
        )

        fig_interaction, ax_interaction = plt.subplots(
            figsize=(10, 6)
        )

        for group, total in group_totals.items():
            data = GROUPS[group]

            fractions = np.array([
                distribution(
                    10 ** (-pH),
                    total,
                    data["pKas"]
                )
                for pH in pH_values
            ])

            for index, species_name in enumerate(
                data["species"]
            ):
                ax_interaction.plot(
                    pH_values,
                    fractions[:, index] * 1000,
                    linewidth=2,
                    label=species_name
                )

        h_values = 10 ** (-pH_values)
        oh_values = 1e-14 / h_values

        ax_interaction.plot(
            pH_values,
            h_values * 1000,
            linestyle="--",
            linewidth=2,
            label="H⁺"
        )

        ax_interaction.plot(
            pH_values,
            oh_values * 1000,
            linestyle="--",
            linewidth=2,
            label="OH⁻"
        )

        ax_interaction.set_title(
            "Equilibrium species concentration versus pH"
        )
        ax_interaction.set_xlabel("pH")
        ax_interaction.set_ylabel(
            "Concentration (mM)"
        )
        ax_interaction.set_xlim(0, 14)
        ax_interaction.grid(alpha=0.3)

        ax_interaction.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False
        )

        fig_interaction.tight_layout()
        st.pyplot(fig_interaction)

        # ----------------------------------------------------
        # Titration graph using selected reagent
        # ----------------------------------------------------

        st.subheader(
            "Titration response of the selected reagent"
        )

        titrant = st.selectbox(
            "Select titrant for the graph",
            selected_reagents
        )

        starting_amount = reagent_amounts[titrant]

        titrant_values = np.linspace(
            0.0,
            max(starting_amount * 2.0, 0.001),
            150
        )

        titration_curve = [
            solve_general(
                selected_reagents,
                reagent_amounts,
                group_totals,
                extra_reagent=titrant,
                extra_amount=value
            )
            for value in titrant_values
        ]

        fig_titration, ax_titration = plt.subplots(
            figsize=(10, 6)
        )

        for species_name in result["species"]:
            values = [
                item["species"].get(
                    species_name,
                    0.0
                ) * 1000
                for item in titration_curve
            ]

            ax_titration.plot(
                titrant_values,
                values,
                linewidth=2,
                label=species_name
            )

        ax_titration.set_title(
            f"Species concentration versus {titrant} concentration"
        )
        ax_titration.set_xlabel(
            f"{titrant} concentration (M)"
        )
        ax_titration.set_ylabel(
            "Concentration (mM)"
        )
        ax_titration.grid(alpha=0.3)

        ax_titration.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False
        )

        fig_titration.tight_layout()
        st.pyplot(fig_titration)

        graph_buffer = io.BytesIO()

        fig_titration.savefig(
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
            "No validation results or graphs are shown "
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


# ============================================================
# ABOUT PAGE
# ============================================================

with about_page:
    st.title("About")

    st.markdown(
        """
        ### Model description

        AquaEquil calculates ideal aqueous acid-base equilibrium
        using electroneutrality and acid dissociation equilibria.

        ### Automatic species selection

        Species groups are generated automatically from the selected
        reagents. The user does not enter separate OCl, CO₃, SO₄,
        or PO₄ concentrations.

        ### Assigned pKa values

        OCl uses pKa 7.53.

        CO₃ uses pKa values 6.35 and 10.33.

        SO₄ uses pKa values -3.00 and 1.99.

        PO₄ uses pKa values 2.15, 7.20, and 12.35.

        ### Assumptions

        - Ideal aqueous solution
        - Constant temperature
        - Constant volume
        - No activity corrections
        - No precipitation
        - No complexation
        - No redox reactions
        """
    )
