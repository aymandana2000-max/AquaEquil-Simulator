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
# ORIGINAL VALIDATION MODEL
# ============================================================

def calculate_original_equilibrium(
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
        "H": h,
        "OH": oh,
        "HOCl": hocl,
        "OCl": ocl,
        "pH": -np.log10(h),
        "charge_residual": charge_balance(log_h),
        "mass_error": hocl + ocl - chlorine_total
    }


# ============================================================
# GENERAL SPECIES MODEL
# ============================================================

SPECIES_GROUPS = {
    "OCl": {
        "species": ["HOCl", "OCl-"],
        "pKas": [7.53],
        "charges": [0, -1]
    },
    "CO3": {
        "species": ["H2CO3", "HCO3-", "CO3--"],
        "pKas": [6.35, 10.33],
        "charges": [0, -1, -2]
    },
    "SO4": {
        "species": ["H2SO4", "HSO4-", "SO4--"],
        "pKas": [-3.0, 1.99],
        "charges": [0, -1, -2]
    },
    "PO4": {
        "species": ["H3PO4", "H2PO4-", "HPO4--", "PO4---"],
        "pKas": [2.15, 7.20, 12.35],
        "charges": [0, -1, -2, -3]
    }
}


REAGENTS = {
    "NaOH": {
        "cation": ("Na+", 1),
        "anion": ("OH-", -1),
        "acid_equivalents": 0
    },
    "KOH": {
        "cation": ("K+", 1),
        "anion": ("OH-", -1),
        "acid_equivalents": 0
    },
    "LiOH": {
        "cation": ("Li+", 1),
        "anion": ("OH-", -1),
        "acid_equivalents": 0
    },
    "Ca(OH)2": {
        "cation": ("Ca++", 2),
        "anion": ("OH-", -1),
        "acid_equivalents": 0
    },
    "HCl": {
        "cation": ("H+", 1),
        "anion": ("Cl-", -1),
        "acid_equivalents": 1
    },
    "HNO3": {
        "cation": ("H+", 1),
        "anion": ("NO3-", -1),
        "acid_equivalents": 1
    },
    "H2SO4": {
        "cation": ("H+", 2),
        "anion": ("SO4--", -2),
        "acid_equivalents": 2
    },
    "NaOCl": {
        "cation": ("Na+", 1),
        "anion": ("OCl-", -1),
        "group": "OCl",
        "group_charge": 1
    },
    "KOCl": {
        "cation": ("K+", 1),
        "anion": ("OCl-", -1),
        "group": "OCl",
        "group_charge": 1
    },
    "Na2CO3": {
        "cation": ("Na+", 2),
        "anion": ("CO3--", -2),
        "group": "CO3",
        "group_charge": 2
    },
    "K2CO3": {
        "cation": ("K+", 2),
        "anion": ("CO3--", -2),
        "group": "CO3",
        "group_charge": 2
    },
    "NaHCO3": {
        "cation": ("Na+", 1),
        "anion": ("HCO3-", -1),
        "group": "CO3",
        "group_charge": 1
    },
    "Na2SO4": {
        "cation": ("Na+", 2),
        "anion": ("SO4--", -2),
        "group": "SO4",
        "group_charge": 2
    },
    "K2SO4": {
        "cation": ("K+", 2),
        "anion": ("SO4--", -2),
        "group": "SO4",
        "group_charge": 2
    },
    "Na3PO4": {
        "cation": ("Na+", 3),
        "anion": ("PO4---", -3),
        "group": "PO4",
        "group_charge": 3
    }
}


def distribution(h, total, pKas):
    kas = 10 ** (-np.asarray(pKas, dtype=float))
    n = len(kas)

    terms = [h ** n]

    for i in range(1, n + 1):
        terms.append(
            np.prod(kas[:i]) * h ** (n - i)
        )

    fractions = np.asarray(terms) / np.sum(terms)
    return total * fractions


def solve_general_equilibrium(
    reagents,
    species_inputs,
    hcl_mM,
    kw
):
    chloride = hcl_mM / 1000
    fixed_positive = 0.0
    fixed_negative = chloride
    families = []

    for reagent in reagents:
        data = REAGENTS[reagent]
        amount = st.session_state.get(
            f"reagent_amount_{reagent}",
            0.0
        )

        cation_name, cation_charge = data["cation"]
        anion_name, anion_charge = data["anion"]

        if cation_charge > 0:
            fixed_positive += amount * cation_charge

        if anion_charge < 0:
            fixed_negative += amount * abs(anion_charge)

        if data.get("acid_equivalents", 0) > 0:
            fixed_positive += amount * data["acid_equivalents"]

    for group_name, values in species_inputs.items():
        total = values["total"]
        pKas = values["pKas"]
        charges = SPECIES_GROUPS[group_name]["charges"]

        families.append({
            "name": group_name,
            "total": total,
            "pKas": pKas,
            "charges": charges,
            "species": SPECIES_GROUPS[group_name]["species"]
        })

        group_charge = max(abs(min(charges)), 1)
        fixed_positive += total * group_charge

    def charge_balance(log_h):
        h = 10 ** log_h
        oh = kw / h
        positive = h + fixed_positive
        negative = fixed_negative + oh

        for family in families:
            values = distribution(
                h,
                family["total"],
                family["pKas"]
            )

            for concentration, charge in zip(
                values,
                family["charges"]
            ):
                if charge < 0:
                    negative += concentration * abs(charge)

        return positive - negative

    log_h = brentq(charge_balance, -14, 0)
    h = 10 ** log_h
    oh = kw / h

    species = {
        "H+": h,
        "OH-": oh
    }

    for family in families:
        values = distribution(
            h,
            family["total"],
            family["pKas"]
        )

        for name, value in zip(
            family["species"],
            values
        ):
            species[name] = value

    return {
        "pH": -np.log10(h),
        "species": species,
        "charge_residual": charge_balance(log_h)
    }


# ============================================================
# SIDEBAR AND MAIN PAGE
# ============================================================

st.title("AquaEquil Chemical Equilibrium Simulator")

st.sidebar.header("General conditions")

kw = st.sidebar.number_input(
    "Water Kw",
    min_value=1e-16,
    value=1e-14,
    format="%.2e"
)

hcl_mM = st.sidebar.number_input(
    "Additional HCl concentration (mM)",
    min_value=0.0,
    value=0.0,
    format="%.4f"
)

st.sidebar.header("Reagents")

selected_reagents = st.sidebar.multiselect(
    "Choose added reagent(s)",
    list(REAGENTS.keys()),
    default=["NaOH"]
)

for reagent in selected_reagents:
    st.sidebar.number_input(
        f"{reagent} amount (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key=f"reagent_amount_{reagent}"
    )

st.sidebar.header("Species groups")

selected_groups = st.sidebar.multiselect(
    "Choose species group(s)",
    list(SPECIES_GROUPS.keys()),
    default=["OCl"]
)

species_inputs = {}

for group in selected_groups:
    group_data = SPECIES_GROUPS[group]

    st.sidebar.markdown(f"### {group}")

    total = st.sidebar.number_input(
        f"{group} total concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key=f"total_{group}"
    )

    pKas = []

    for index, default in enumerate(group_data["pKas"]):
        pka = st.sidebar.number_input(
            f"{group} pKa {index + 1}",
            value=float(default),
            format="%.4f",
            key=f"pka_{group}_{index}"
        )

        pKas.append(pka)

    species_inputs[group] = {
        "total": total,
        "pKas": pKas
    }

if not selected_reagents:
    st.warning("Choose at least one reagent.")
    st.stop()

if not selected_groups:
    st.warning("Choose at least one species group.")
    st.stop()

result = solve_general_equilibrium(
    selected_reagents,
    species_inputs,
    hcl_mM,
    kw
)


# ============================================================
# TABS
# ============================================================

main_tab, validation_tab, graph_tab, titration_tab = st.tabs(
    [
        "Main",
        "Validation",
        "Independent Graphs",
        "Titration Graph"
    ]
)


# ============================================================
# MAIN PAGE
# ============================================================

with main_tab:
    st.subheader("Selected Chemical System")

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

    st.markdown("### All Equilibrium Species")

    result_table = pd.DataFrame({
        "Species": list(result["species"].keys()),
        "Concentration (M)": list(result["species"].values())
    })

    st.dataframe(
        result_table.style.format({
            "Concentration (M)": "{:.6e}"
        }),
        use_container_width=True
    )

    st.markdown(
        f"Charge residual: "
        f"`{result['charge_residual']:.6e} M`"
    )


# ============================================================
# VALIDATION PAGE
# ============================================================

with validation_tab:
    st.subheader("Original NaOH–NaOCl–HCl Validation")

    st.info(
        "This page preserves the original validation cases "
        "and does not use the new general reagent model."
    )

    naoh_validation = st.number_input(
        "Validation NaOH concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key="validation_naoh"
    )

    naocl_validation = st.number_input(
        "Validation NaOCl concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key="validation_naocl"
    )

    pka_validation = st.number_input(
        "Validation HOCl pKa",
        value=7.53,
        format="%.4f",
        key="validation_pka"
    )

    validation_hcl = [
        0.00,
        0.50,
        1.00,
        1.50,
        1.90
    ]

    reference_pH = {
        0.00: 11.00,
        0.50: 10.70,
        1.00: 9.26,
        1.50: 7.53,
        1.90: 6.58
    }

    rows = []

    for hcl in validation_hcl:
        current = calculate_original_equilibrium(
            naoh_validation,
            naocl_validation,
            hcl,
            pka_validation,
            kw
        )

        expected = reference_pH[hcl]
        difference = current["pH"] - expected

        rows.append({
            "HCl (mM)": hcl,
            "Calculated pH": current["pH"],
            "Expected pH": expected,
            "Difference": difference,
            "Charge residual (M)": current["charge_residual"],
            "Mass error (M)": current["mass_error"],
            "Status": (
                "PASS"
                if abs(difference) <= 0.10
                and abs(current["charge_residual"]) < 1e-7
                and abs(current["mass_error"]) < 1e-7
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

    st.download_button(
        "Download Validation CSV",
        validation_df.to_csv(index=False),
        "AquaEquil_validation.csv",
        "text/csv"
    )

    st.markdown("### Original Validation Graphs")

    validation_curve = [
        calculate_original_equilibrium(
            naoh_validation,
            naocl_validation,
            hcl,
            pka_validation,
            kw
        )
        for hcl in np.linspace(0, 1.9, 100)
    ]

    hcl_curve = np.linspace(0, 1.9, 100)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8)
    )

    axes[0].plot(
        hcl_curve,
        [x["pH"] for x in validation_curve],
        linewidth=2,
        label="Calculated pH"
    )

    axes[0].scatter(
        validation_hcl,
        [
            calculate_original_equilibrium(
                naoh_validation,
                naocl_validation,
                hcl,
                pka_validation,
                kw
            )["pH"]
            for hcl in validation_hcl
        ],
        color="red",
        marker="x",
        label="Validation points"
    )

    axes[0].set_title("Original Validation Titration Curve")
    axes[0].set_xlabel("HCl addition (mM)")
    axes[0].set_ylabel("pH")
    axes[0].set_ylim(0, 14)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        hcl_curve,
        [x["HOCl"] * 1000 for x in validation_curve],
        label="HOCl",
        linewidth=2
    )

    axes[1].plot(
        hcl_curve,
        [x["OCl"] * 1000 for x in validation_curve],
        label="OCl-",
        linewidth=2
    )

    axes[1].set_title("Original HOCl/OCl- Speciation")
    axes[1].set_xlabel("HCl addition (mM)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig)


# ============================================================
# INDEPENDENT GRAPH PAGE
# ============================================================

with graph_tab:
    st.subheader("Independent Graphs")

    st.write(
        "These graphs represent only the reagents and species "
        "selected on the Main page."
    )

    hcl_values = np.linspace(
        0,
        max(1.9, hcl_mM * 1.5 + 0.1),
        120
    )

    curve = [
        solve_general_equilibrium(
            selected_reagents,
            species_inputs,
            hcl_value,
            kw
        )
        for hcl_value in hcl_values
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 14)
    )

    axes[0].plot(
        hcl_values,
        [x["pH"] for x in curve],
        color="blue",
        linewidth=2
    )

    axes[0].set_title("pH of Selected Reaction System")
    axes[0].set_xlabel("HCl addition (mM)")
    axes[0].set_ylabel("pH")
    axes[0].set_ylim(0, 14)
    axes[0].grid(alpha=0.3)

    species_names = list(result["species"].keys())

    for name in species_names:
        axes[1].plot(
            hcl_values,
            [
                x["species"].get(name, 0) * 1000
                for x in curve
            ],
            linewidth=2,
            label=name
        )

    axes[1].set_title("All Interacting Species")
    axes[1].set_xlabel("HCl addition (mM)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3)

    for group in selected_groups:
        group_data = SPECIES_GROUPS[group]
        pH_values = np.linspace(0, 14, 250)

        fractions = np.array([
            distribution(
                10 ** (-pH),
                1.0,
                species_inputs[group]["pKas"]
            )
            for pH in pH_values
        ])

        for index, name in enumerate(group_data["species"]):
            axes[2].plot(
                pH_values,
                fractions[:, index] * 100,
                linewidth=2,
                label=name
            )

    axes[2].set_title("Fractional Species Distribution")
    axes[2].set_xlabel("pH")
    axes[2].set_ylabel("Fraction (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=3)

    fig.tight_layout()
    st.pyplot(fig)


# ============================================================
# TITRATION PAGE
# ============================================================

with titration_tab:
    st.subheader("Titration Graph")

    hcl_values = np.linspace(
        0,
        max(1.9, hcl_mM * 1.5 + 0.1),
        150
    )

    curve = [
        solve_general_equilibrium(
            selected_reagents,
            species_inputs,
            hcl_value,
            kw
        )
        for hcl_value in hcl_values
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    plotted_names = set()

    for group in selected_groups:
        group_data = SPECIES_GROUPS[group]

        for index, name in enumerate(group_data["species"]):
            values = []

            for item in curve:
                values.append(
                    distribution(
                        10 ** (-item["pH"]),
                        1.0,
                        species_inputs[group]["pKas"]
                    )[index] * 100
                )

            ax.plot(
                hcl_values,
                values,
                linewidth=2,
                label=f"{name} fraction"
            )

            plotted_names.add(name)

    ax.set_title(
        "Species Start and End Markers During Titration"
    )

    ax.set_xlabel("HCl addition (mM)")
    ax.set_ylabel("Fraction of family (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Markers occur at pKa transition points.
    # The plotted marker is placed at the HCl amount whose pH is
    # closest to each pKa.
    marker_hcl = []

    for group in selected_groups:
        for pka in species_inputs[group]["pKas"]:
            index = np.argmin([
                abs(item["pH"] - pka)
                for item in curve
            ])

            x_value = hcl_values[index]
            marker_hcl.append(x_value)

            ax.axvline(
                x_value,
                color="black",
                linestyle="--",
                alpha=0.45
            )

            ax.text(
                x_value,
                96,
                f"pKa {pka:.2f}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )

    ax.legend(ncol=3)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown(
        "Dashed vertical lines mark approximate pKa transition locations. "
        "They show where adjacent protonation states become comparable."
    )
