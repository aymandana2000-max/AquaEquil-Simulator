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
# DISPLAY LABELS
# ============================================================

DISPLAY = {
    "OCl": ["HOCl", "OCl⁻"],
    "CO3": ["H₂CO₃", "HCO₃⁻", "CO₃²⁻"],
    "SO4": ["H₂SO₄", "HSO₄⁻", "SO₄²⁻"],
    "PO4": ["H₃PO₄", "H₂PO₄⁻", "HPO₄²⁻", "PO₄³⁻"],
    "NaOH": "NaOH",
    "KOH": "KOH",
    "LiOH": "LiOH",
    "CaOH2": "Ca(OH)₂",
    "HCl": "HCl",
    "HNO3": "HNO₃",
    "H2SO4": "H₂SO₄",
    "NaOCl": "NaOCl",
    "KOCl": "KOCl",
    "Na2CO3": "Na₂CO₃",
    "K2CO3": "K₂CO₃",
    "NaHCO3": "NaHCO₃",
    "Na2SO4": "Na₂SO₄",
    "K2SO4": "K₂SO₄",
    "Na3PO4": "Na₃PO₄"
}


# ============================================================
# SPECIES GROUPS
# ============================================================

GROUPS = {
    "OCl": {
        "species": ["HOCl", "OCl⁻"],
        "pKas": [7.53],
        "charges": [0, -1]
    },
    "CO3": {
        "species": ["H₂CO₃", "HCO₃⁻", "CO₃²⁻"],
        "pKas": [6.35, 10.33],
        "charges": [0, -1, -2]
    },
    "SO4": {
        "species": ["H₂SO₄", "HSO₄⁻", "SO₄²⁻"],
        "pKas": [-3.0, 1.99],
        "charges": [0, -1, -2]
    },
    "PO4": {
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
        "display": "NaOH",
        "groups": [],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "KOH": {
        "display": "KOH",
        "groups": [],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "LiOH": {
        "display": "LiOH",
        "groups": [],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "CaOH2": {
        "display": "Ca(OH)₂",
        "groups": [],
        "positive_charge": 2,
        "negative_charge": 2,
        "acid_equivalents": 0
    },
    "HCl": {
        "display": "HCl",
        "groups": [],
        "positive_charge": 0,
        "negative_charge": 1,
        "acid_equivalents": 1
    },
    "HNO3": {
        "display": "HNO₃",
        "groups": [],
        "positive_charge": 0,
        "negative_charge": 1,
        "acid_equivalents": 1
    },
    "H2SO4": {
        "display": "H₂SO₄",
        "groups": ["SO4"],
        "positive_charge": 0,
        "negative_charge": 2,
        "acid_equivalents": 2
    },
    "NaOCl": {
        "display": "NaOCl",
        "groups": ["OCl"],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "KOCl": {
        "display": "KOCl",
        "groups": ["OCl"],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "Na2CO3": {
        "display": "Na₂CO₃",
        "groups": ["CO3"],
        "positive_charge": 2,
        "negative_charge": 2,
        "acid_equivalents": 0
    },
    "K2CO3": {
        "display": "K₂CO₃",
        "groups": ["CO3"],
        "positive_charge": 2,
        "negative_charge": 2,
        "acid_equivalents": 0
    },
    "NaHCO3": {
        "display": "NaHCO₃",
        "groups": ["CO3"],
        "positive_charge": 1,
        "negative_charge": 1,
        "acid_equivalents": 0
    },
    "Na2SO4": {
        "display": "Na₂SO₄",
        "groups": ["SO4"],
        "positive_charge": 2,
        "negative_charge": 2,
        "acid_equivalents": 0
    },
    "K2SO4": {
        "display": "K₂SO₄",
        "groups": ["SO4"],
        "positive_charge": 2,
        "negative_charge": 2,
        "acid_equivalents": 0
    },
    "Na3PO4": {
        "display": "Na₃PO₄",
        "groups": ["PO4"],
        "positive_charge": 3,
        "negative_charge": 3,
        "acid_equivalents": 0
    }
}


# ============================================================
# EQUILIBRIUM FUNCTIONS
# ============================================================

def distribution(h, total, pKas):
    kas = 10 ** (-np.asarray(pKas, dtype=float))
    n = len(kas)

    terms = [h ** n]

    for i in range(1, n + 1):
        terms.append(
            np.prod(kas[:i]) * h ** (n - i)
        )

    terms = np.asarray(terms)
    return total * terms / terms.sum()


def solve_general(reagents, reagent_amounts, groups, kw):
    fixed_positive = 0.0
    fixed_negative = 0.0
    families = []

    for reagent in reagents:
        amount = reagent_amounts.get(reagent, 0.0)
        info = REAGENTS[reagent]

        fixed_positive += amount * info["positive_charge"]
        fixed_negative += amount * info["negative_charge"]

        if info["acid_equivalents"]:
            fixed_positive += amount * info["acid_equivalents"]

    for group_name, settings in groups.items():
        data = GROUPS[group_name]
        total = settings["total"]
        pKas = settings["pKas"]
        charges = data["charges"]

        families.append({
            "name": group_name,
            "total": total,
            "pKas": pKas,
            "charges": charges,
            "species": data["species"]
        })

        maximum_charge = abs(min(charges))
        fixed_positive += total * maximum_charge

    def charge_balance(log_h):
        h = 10 ** log_h
        oh = kw / h

        positive = h + fixed_positive
        negative = oh + fixed_negative

        for family in families:
            values = distribution(
                h,
                family["total"],
                family["pKas"]
            )

            for value, charge in zip(
                values,
                family["charges"]
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
# ORIGINAL VALIDATION FUNCTION
# ============================================================

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

    def balance(log_h):
        h = 10 ** log_h
        oh = kw / h
        ocl = chlorine_total * ka / (ka + h)

        return h + sodium - oh - ocl - chloride

    log_h = brentq(balance, -14, 0)
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
        "charge_residual": balance(log_h),
        "mass_error": hocl + ocl - chlorine_total
    }


# ============================================================
# MAIN PAGE INPUTS
# ============================================================

st.title("AquaEquil Chemical Equilibrium Simulator")

st.sidebar.header("Reagents")

selected_reagents = st.sidebar.multiselect(
    "Select reagent(s)",
    list(REAGENTS.keys()),
    default=["NaOCl"]
)

reagent_amounts = {}

for reagent in selected_reagents:
    reagent_amounts[reagent] = st.sidebar.number_input(
        f"{REAGENTS[reagent]['display']} amount (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key=f"amount_{reagent}"
    )

st.sidebar.header("Species Groups")

if len(selected_reagents) == 1:
    related_groups = set(
        REAGENTS[selected_reagents[0]]["groups"]
    )

    visible_groups = [
        group for group in GROUPS
        if group in related_groups
    ]
else:
    visible_groups = list(GROUPS.keys())

selected_groups = st.sidebar.multiselect(
    "Select species group(s)",
    visible_groups,
    default=visible_groups[:1]
)

group_settings = {}

for group in selected_groups:
    st.sidebar.markdown(f"### {group}")

    total = st.sidebar.number_input(
        f"{group} total concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key=f"total_{group}"
    )

    pKas = []

    for index, default in enumerate(GROUPS[group]["pKas"]):
        pKas.append(
            st.sidebar.number_input(
                f"{group} pKa {index + 1}",
                value=float(default),
                format="%.4f",
                key=f"pka_{group}_{index}"
            )
        )

    group_settings[group] = {
        "total": total,
        "pKas": pKas
    }

if not selected_reagents:
    st.warning("Select at least one reagent.")
    st.stop()

if not selected_groups:
    st.warning("Select at least one species group.")
    st.stop()

kw = 1e-14

result = solve_general(
    selected_reagents,
    reagent_amounts,
    group_settings,
    kw
)


# ============================================================
# MAIN PAGE AND VISUALIZATION
# ============================================================

main_tab, validation_tab = st.tabs(
    [
        "Main / Visualization",
        "Validation"
    ]
)


with main_tab:
    st.subheader("Current Equilibrium")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Equilibrium pH",
            f"{result['pH']:.5f}"
        )

    with col2:
        dominant_species = max(
            result["species"],
            key=result["species"].get
        )

        st.metric(
            "Largest Species",
            dominant_species
        )

    st.subheader("Equilibrium Species")

    species_df = pd.DataFrame({
        "Species": list(result["species"].keys()),
        "Concentration (M)": list(result["species"].values())
    })

    st.dataframe(
        species_df.style.format({
            "Concentration (M)": "{:.6e}"
        }),
        use_container_width=True
    )

    st.markdown(
        f"Charge residual: "
        f"`{result['charge_residual']:.6e} M`"
    )

    st.divider()
    st.subheader("Visualization")

    hcl_values = np.linspace(0, 2.0, 120)

    curves = [
        solve_general(
            selected_reagents,
            reagent_amounts,
            group_settings,
            kw
        )
        for _ in hcl_values
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10)
    )

    axes[0].plot(
        hcl_values,
        [item["pH"] for item in curves],
        color="blue",
        linewidth=2
    )

    axes[0].set_title("pH Response of Selected Reaction")
    axes[0].set_xlabel("HCl addition (mM)")
    axes[0].set_ylabel("pH")
    axes[0].set_ylim(0, 14)
    axes[0].grid(alpha=0.3)

    for name in result["species"]:
        axes[1].plot(
            hcl_values,
            [
                item["species"].get(name, 0) * 1000
                for item in curves
            ],
            linewidth=2,
            label=name
        )

    axes[1].set_title("All Interacting Species")
    axes[1].set_xlabel("HCl addition (mM)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3)

    fig.tight_layout()
    st.pyplot(fig)

    graph_buffer = io.BytesIO()

    fig.savefig(
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

with validation_tab:
    st.subheader("Original Validation")

    st.info(
        "Enter the original validation concentrations, "
        "then press Run Validation."
    )

    validation_naoh = st.number_input(
        "NaOH concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key="validation_naoh"
    )

    validation_naocl = st.number_input(
        "NaOCl concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key="validation_naocl"
    )

    validation_pka = st.number_input(
        "HOCl pKa",
        value=7.53,
        format="%.4f",
        key="validation_pka"
    )

    validation_kw = st.number_input(
        "Kw",
        min_value=1e-16,
        value=1e-14,
        format="%.2e",
        key="validation_kw"
    )

    validation_hcl_max = st.number_input(
        "Maximum HCl concentration (mM)",
        min_value=0.01,
        value=1.90,
        format="%.4f",
        key="validation_hcl_max"
    )

    run_validation = st.button(
        "Run Validation",
        type="primary"
    )

    if run_validation:
        hcl_cases = [
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

        for hcl in hcl_cases:
            current = original_equilibrium(
                validation_naoh,
                validation_naocl,
                hcl,
                validation_pka,
                validation_kw
            )

            difference = current["pH"] - expected_pH[hcl]

            passed = (
                abs(difference) <= 0.10
                and abs(current["charge_residual"]) < 1e-7
                and abs(current["mass_error"]) < 1e-7
            )

            rows.append({
                "HCl (mM)": hcl,
                "Calculated pH": current["pH"],
                "Expected pH": expected_pH[hcl],
                "Difference": difference,
                "Charge residual (M)": current["charge_residual"],
                "Mass error (M)": current["mass_error"],
                "Status": "PASS" if passed else "CHECK"
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

        hcl_curve = np.linspace(
            0,
            validation_hcl_max,
            120
        )

        validation_curve = [
            original_equilibrium(
                validation_naoh,
                validation_naocl,
                value,
                validation_pka,
                validation_kw
            )
            for value in hcl_curve
        ]

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
            hcl_cases,
            [
                original_equilibrium(
                    validation_naoh,
                    validation_naocl,
                    value,
                    validation_pka,
                    validation_kw
                )["pH"]
                for value in hcl_cases
            ],
            color="red",
            marker="x",
            s=70,
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
            label="OCl⁻",
            linewidth=2
        )

        axes[1].set_title("Original HOCl/OCl⁻ Speciation")
        axes[1].set_xlabel("HCl addition (mM)")
        axes[1].set_ylabel("Concentration (mM)")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        st.pyplot(fig)

        validation_buffer = io.BytesIO()

        fig.savefig(
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
