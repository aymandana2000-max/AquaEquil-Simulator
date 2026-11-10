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
# CHEMICAL LABELS
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
        "pKas": [-3.0, 1.99],
        "charges": [0, -1, -2]
    },
    "PO₄": {
        "species": ["H₃PO₄", "H₂PO₄⁻", "HPO₄²⁻", "PO₄³⁻"],
        "pKas": [2.15, 7.20, 12.35],
        "charges": [0, -1, -2, -3]
    }
}


REAGENTS = {
    "NaOH": {
        "label": "NaOH",
        "groups": [],
        "positive": 1,
        "negative": 1
    },
    "KOH": {
        "label": "KOH",
        "groups": [],
        "positive": 1,
        "negative": 1
    },
    "LiOH": {
        "label": "LiOH",
        "groups": [],
        "positive": 1,
        "negative": 1
    },
    "Ca(OH)₂": {
        "label": "Ca(OH)₂",
        "groups": [],
        "positive": 2,
        "negative": 2
    },
    "HCl": {
        "label": "HCl",
        "groups": [],
        "positive": 0,
        "negative": 1
    },
    "HNO₃": {
        "label": "HNO₃",
        "groups": [],
        "positive": 0,
        "negative": 1
    },
    "H₂SO₄": {
        "label": "H₂SO₄",
        "groups": ["SO₄"],
        "positive": 0,
        "negative": 2
    },
    "NaOCl": {
        "label": "NaOCl",
        "groups": ["OCl"],
        "positive": 1,
        "negative": 1
    },
    "KOCl": {
        "label": "KOCl",
        "groups": ["OCl"],
        "positive": 1,
        "negative": 1
    },
    "Na₂CO₃": {
        "label": "Na₂CO₃",
        "groups": ["CO₃"],
        "positive": 2,
        "negative": 2
    },
    "K₂CO₃": {
        "label": "K₂CO₃",
        "groups": ["CO₃"],
        "positive": 2,
        "negative": 2
    },
    "NaHCO₃": {
        "label": "NaHCO₃",
        "groups": ["CO₃"],
        "positive": 1,
        "negative": 1
    },
    "Na₂SO₄": {
        "label": "Na₂SO₄",
        "groups": ["SO₄"],
        "positive": 2,
        "negative": 2
    },
    "K₂SO₄": {
        "label": "K₂SO₄",
        "groups": ["SO₄"],
        "positive": 2,
        "negative": 2
    },
    "Na₃PO₄": {
        "label": "Na₃PO₄",
        "groups": ["PO₄"],
        "positive": 3,
        "negative": 3
    }
}


# ============================================================
# GENERAL EQUILIBRIUM MODEL
# ============================================================

def distribution(h, total, pKas):
    ka_values = 10 ** (-np.asarray(pKas, dtype=float))
    number_of_steps = len(ka_values)

    terms = [h ** number_of_steps]

    for i in range(1, number_of_steps + 1):
        terms.append(
            np.prod(ka_values[:i])
            * h ** (number_of_steps - i)
        )

    terms = np.asarray(terms)

    return total * terms / terms.sum()


def solve_general(
    selected_reagents,
    reagent_amounts,
    selected_groups,
    group_inputs,
    kw=1e-14
):
    fixed_positive = 0.0
    fixed_negative = 0.0
    families = []

    for reagent in selected_reagents:
        amount = reagent_amounts[reagent]
        reagent_data = REAGENTS[reagent]

        fixed_positive += (
            amount * reagent_data["positive"]
        )

        fixed_negative += (
            amount * reagent_data["negative"]
        )

    for group_name in selected_groups:
        group_data = GROUPS[group_name]
        settings = group_inputs[group_name]

        total = settings["total"]
        pKas = settings["pKas"]

        families.append({
            "name": group_name,
            "species": group_data["species"],
            "charges": group_data["charges"],
            "total": total,
            "pKas": pKas
        })

        maximum_negative_charge = abs(
            min(group_data["charges"])
        )

        fixed_positive += (
            total * maximum_negative_charge
        )

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

            for concentration, charge in zip(
                amounts,
                family["charges"]
            ):
                if charge < 0:
                    negative += (
                        concentration * abs(charge)
                    )

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


# ============================================================
# ORIGINAL VALIDATION MODEL
# ============================================================

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
# MAIN PAGE
# ============================================================

st.title("Main")

st.sidebar.header("Reagents")

selected_reagents = st.sidebar.multiselect(
    "Select reagent(s)",
    list(REAGENTS.keys()),
    default=[]
)

reagent_amounts = {}

for reagent in selected_reagents:
    reagent_amounts[reagent] = st.sidebar.number_input(
        f"{REAGENTS[reagent]['label']} amount (M)",
        min_value=0.0,
        value=None,
        step=0.1,
        format="%.5f",
        key=f"amount_{reagent}"
    )


if len(selected_reagents) == 1:
    reagent_groups = set(
        REAGENTS[selected_reagents[0]]["groups"]
    )

    if reagent_groups:
        visible_groups = [
            group for group in GROUPS
            if group in reagent_groups
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

    pKas = []

    for index, default_pka in enumerate(
        GROUPS[group]["pKas"]
    ):
        pka = st.sidebar.number_input(
            f"{group} pKa {index + 1}",
            min_value=-20.0,
            max_value=20.0,
            value=None,
            step=0.1,
            format="%.1f",
            key=f"pka_{group}_{index}"
        )

        pKas.append(pka)

    group_inputs[group] = {
        "total": total,
        "pKas": pKas
    }


# ============================================================
# MAIN PAGE STATUS
# ============================================================

missing_reagent_amounts = [
    reagent
    for reagent in selected_reagents
    if reagent_amounts[reagent] is None
]

missing_species_values = [
    group
    for group in selected_groups
    if group_inputs[group]["total"] is None
    or any(
        pka is None
        for pka in group_inputs[group]["pKas"]
    )
]


if (
    not selected_reagents
    or not selected_groups
    or missing_reagent_amounts
    or missing_species_values
):
    st.info(
        "Choose reagent(s) and species group(s), "
        "then enter all concentrations and pKa values."
    )
    st.stop()


result = solve_general(
    selected_reagents,
    reagent_amounts,
    selected_groups,
    group_inputs
)


# ============================================================
# MAIN RESULTS
# ============================================================

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

st.subheader("Equilibrium Species")

species_df = pd.DataFrame({
    "Species": list(result["species"].keys()),
    "Concentration (M)": list(result["species"].values())
})

st.dataframe(
    species_df.style.format({
        "Concentration (M)": "{:.5f}"
    }),
    use_container_width=True
)

st.write(
    "Charge residual:",
    f"{result['charge_residual']:.6e} M"
)


# ============================================================
# VISUALIZATION
# ============================================================

st.header("Visualization")

hcl_values = np.linspace(0, 1.9, 120)

curve = [
    solve_general(
        selected_reagents,
        reagent_amounts,
        selected_groups,
        group_inputs
    )
    for _ in hcl_values
]


st.subheader("pH versus HCl addition")

fig_pH, ax_pH = plt.subplots(
    figsize=(10, 5)
)

ax_pH.plot(
    hcl_values,
    [item["pH"] for item in curve],
    linewidth=2,
    label="Calculated pH"
)

validation_hcl_points = [
    0.00,
    0.50,
    1.00,
    1.50,
    1.90
]

if (
    "NaOH" in selected_reagents
    and "NaOCl" in selected_reagents
):
    validation_points = [
        original_equilibrium(
            reagent_amounts["NaOH"],
            reagent_amounts["NaOCl"],
            hcl,
            7.53,
            1e-14
        )["pH"]
        for hcl in validation_hcl_points
    ]

    ax_pH.scatter(
        validation_hcl_points,
        validation_points,
        color="red",
        marker="x",
        s=70,
        linewidths=2,
        label="Validation points"
    )

ax_pH.set_title("pH 7 Visualization")
ax_pH.set_xlabel("HCl addition (mM)")
ax_pH.set_ylabel("pH")
ax_pH.set_ylim(0, 14)
ax_pH.grid(alpha=0.3)
ax_pH.legend()

fig_pH.tight_layout()
st.pyplot(fig_pH)


st.subheader("Species concentration versus HCl addition")

fig_species, ax_species = plt.subplots(
    figsize=(10, 6)
)

for species_name in result["species"]:
    ax_species.plot(
        hcl_values,
        [
            item["species"].get(species_name, 0)
            * 1000
            for item in curve
        ],
        linewidth=2,
        label=species_name
    )

ax_species.set_title(
    "Concentration 10⁷ Visualization"
)
ax_species.set_xlabel("HCl addition (mM)")
ax_species.set_ylabel("Concentration (mM)")
ax_species.grid(alpha=0.3)
ax_species.legend(ncol=3)

fig_species.tight_layout()
st.pyplot(fig_species)


visualization_buffer = io.BytesIO()

fig_species.savefig(
    visualization_buffer,
    format="png",
    dpi=300,
    bbox_inches="tight"
)

st.download_button(
    "Download Visualization Graphs as PNG",
    visualization_buffer.getvalue(),
    "AquaEquil_visualization.png",
    "image/png"
)


# ============================================================
# VALIDATION PAGE
# ============================================================

st.divider()
st.header("Validation")

st.info(
    "Enter the original validation values and press "
    "Run Validation. No validation values or graphs are "
    "shown before that button is pressed."
)

validation_naoh = st.number_input(
    "Validation NaOH concentration (M)",
    min_value=0.0,
    value=None,
    step=0.1,
    format="%.5f",
    key="validation_naoh"
)

validation_naocl = st.number_input(
    "Validation NaOCl concentration (M)",
    min_value=0.0,
    value=None,
    step=0.1,
    format="%.5f",
    key="validation_naocl"
)

validation_pka = st.number_input(
    "Validation HOCl pKa",
    min_value=-20.0,
    max_value=20.0,
    value=None,
    step=0.1,
    format="%.1f",
    key="validation_pka"
)

validation_kw = st.number_input(
    "Validation Kw",
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
        "Validation results and graphs will appear "
        "after Run Validation is pressed."
    )
    st.stop()

if any(
    value is None
    for value in [
        validation_naoh,
        validation_naocl,
        validation_pka,
        validation_kw
    ]
):
    st.warning(
        "Enter all validation values before running validation."
    )
    st.stop()


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

validation_rows = []

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
        and abs(current["charge_residual"]) < 1e-7
        and abs(current["mass_error"]) < 1e-7
    )

    validation_rows.append({
        "HCl (mM)": hcl,
        "Calculated pH": current["pH"],
        "Expected pH": expected_pH[hcl],
        "Difference": difference,
        "Charge residual (M)": current["charge_residual"],
        "Mass error (M)": current["mass_error"],
        "Status": "PASS" if passed else "CHECK"
    })

validation_df = pd.DataFrame(validation_rows)

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


st.subheader("Original Validation Graphs")

validation_x = np.linspace(0, 1.9, 120)

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
    [item["pH"] for item in validation_curve],
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

axes[0].set_title("Original Validation Titration Curve")
axes[0].set_xlabel("HCl addition (mM)")
axes[0].set_ylabel("pH")
axes[0].set_ylim(0, 14)
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(
    validation_x,
    [item["HOCl"] * 1000 for item in validation_curve],
    linewidth=2,
    label="HOCl"
)

axes[1].plot(
    validation_x,
    [item["OCl"] * 1000 for item in validation_curve],
    linewidth=2,
    label="OCl⁻"
)

axes[1].set_title("Original HOCl/OCl⁻ Speciation")
axes[1].set_xlabel("HCl addition (mM)")
axes[1].set_ylabel("Concentration (mM)")
axes[1].grid(alpha=0.3)
axes[1].legend()

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
