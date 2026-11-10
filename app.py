
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq

st.set_page_config(page_title="AquaEquil Simulator", layout="wide")


def family_distribution(h, total, pKas):
    kas = 10.0 ** (-np.asarray(pKas, dtype=float))
    n = len(kas)

    terms = [h ** n]

    for i in range(1, n + 1):
        terms.append(np.prod(kas[:i]) * h ** (n - i))

    fractions = np.asarray(terms) / np.sum(terms)
    return total * fractions


def calculate_equilibrium(naoh, hcl_mM, families, kw=1e-14):
    hcl = hcl_mM / 1000.0

    sodium = naoh + sum(
        family["total"] * len(family["pKas"])
        for family in families
    )

    chloride = hcl

    def charge_balance(log_h):
        h = 10.0 ** log_h
        oh = kw / h

        negative_species = oh + chloride

        for family in families:
            states = family_distribution(
                h,
                family["total"],
                family["pKas"]
            )

            negative_species += sum(
                i * states[i]
                for i in range(len(states))
            )

        return h + sodium - negative_species

    log_h = brentq(charge_balance, -14, 0)
    h = 10.0 ** log_h
    oh = kw / h

    species = {
        "H+": h,
        "OH-": oh
    }

    for family in families:
        states = family_distribution(
            h,
            family["total"],
            family["pKas"]
        )

        for name, concentration in zip(
            family["names"],
            states
        ):
            species[name] = concentration

    mass_errors = {}

    for family in families:
        mass_errors[family["label"]] = (
            sum(
                species[name]
                for name in family["names"]
            )
            - family["total"]
        )

    return {
        "H+": h,
        "OH-": oh,
        "pH": -np.log10(h),
        "species": species,
        "charge_residual": charge_balance(log_h),
        "mass_errors": mass_errors
    }


GROUPS = {
    "OCl (hypochlorite)": {
        "names": ["HOCl", "OCl-"],
        "default_pKas": [7.53]
    },
    "CO3 (carbonate)": {
        "names": ["H2CO3", "HCO3-", "CO3--"],
        "default_pKas": [6.35, 10.33]
    },
    "SO4 (sulfate)": {
        "names": ["H2SO4", "HSO4-", "SO4--"],
        "default_pKas": [-3.0, 1.99]
    },
    "PO4 (phosphate)": {
        "names": [
            "H3PO4",
            "H2PO4-",
            "HPO4--",
            "PO4---"
        ],
        "default_pKas": [2.15, 7.20, 12.35]
    }
}


st.title("AquaEquil: Chemical Equilibrium Simulator")
st.caption(
    "General multi-family acid-base equilibrium, validation, and speciation"
)


st.sidebar.header("Solution inputs")

naoh = st.sidebar.number_input(
    "NaOH concentration (M)",
    min_value=0.0,
    value=0.001,
    format="%.8f"
)

hcl_mM = st.sidebar.number_input(
    "HCl concentration (mM)",
    min_value=0.0,
    value=0.0,
    format="%.4f"
)

kw = st.sidebar.number_input(
    "Water Kw",
    min_value=0.0,
    value=1e-14,
    format="%.2e"
)


st.sidebar.header("Acid families")

selected = st.sidebar.multiselect(
    "Select species groups",
    list(GROUPS.keys()),
    default=["OCl (hypochlorite)"]
)


families = []

for label in selected:
    info = GROUPS[label]

    st.sidebar.markdown(f"**{label}**")

    total = st.sidebar.number_input(
        f"{label} total concentration (M)",
        min_value=0.0,
        value=0.001,
        format="%.8f",
        key=f"total_{label}"
    )

    pKas = []

    for i, default in enumerate(info["default_pKas"]):
        pka = st.sidebar.number_input(
            f"{label} pKa {i + 1}",
            value=float(default),
            format="%.4f",
            key=f"pka_{label}_{i}"
        )

        pKas.append(pka)

    families.append({
        "label": label,
        "total": total,
        "pKas": pKas,
        "names": info["names"]
    })


if not families:
    st.warning("Select at least one species group.")

    families = [{
        "label": "OCl (hypochlorite)",
        "total": 0.0,
        "pKas": [7.53],
        "names": ["HOCl", "OCl-"]
    }]


calculate_button = st.sidebar.button(
    "Calculate Equilibrium",
    type="primary"
)


if "has_calculated" not in st.session_state:
    st.session_state.has_calculated = False

if calculate_button:
    st.session_state.has_calculated = True

if not st.session_state.has_calculated:
    st.info(
        "Showing the default calculation. "
        "Change inputs and press Calculate Equilibrium to update it."
    )


result = calculate_equilibrium(
    naoh,
    hcl_mM,
    families,
    kw
)

all_species = result["species"]

dominant_species = max(
    all_species,
    key=all_species.get
)


calculator_tab, validation_tab, graphs_tab, about_tab = st.tabs(
    [
        "Equilibrium Calculator",
        "Validation",
        "Graphs",
        "About the Model"
    ]
)


with calculator_tab:
    st.subheader("Equilibrium Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Equilibrium pH",
            f"{result['pH']:.5f}"
        )

    with col2:
        st.metric(
            "Dominant Species",
            dominant_species
        )

    with col3:
        st.metric(
            "HCl Input",
            f"{hcl_mM:.5f} mM"
        )

    result_table = pd.DataFrame({
        "Species": list(all_species.keys()),
        "Concentration (M)": list(all_species.values())
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

    maximum_mass_error = max(
        abs(value)
        for value in result["mass_errors"].values()
    )

    st.markdown(
        f"Maximum family mass error: "
        f"`{maximum_mass_error:.6e} M`"
    )

    if (
        abs(result["charge_residual"]) < 1e-7
        and maximum_mass_error < 1e-7
    ):
        st.success("Mass and charge balances are satisfied.")
    else:
        st.warning("Check the balance errors.")


with validation_tab:
    st.subheader("Balance Validation")

    validation_rows = []

    for family, error in result["mass_errors"].items():
        validation_rows.append({
            "Family": family,
            "Error (M)": error
        })

    validation_rows.append({
        "Family": "Charge balance",
        "Error (M)": result["charge_residual"]
    })

    validation_df = pd.DataFrame(validation_rows)

    st.dataframe(
        validation_df.style.format({
            "Error (M)": "{:.6e}"
        }),
        use_container_width=True
    )

    st.download_button(
        "Download Validation Results as CSV",
        validation_df.to_csv(index=False),
        "AquaEquil_validation_results.csv",
        "text/csv"
    )


with graphs_tab:
    st.subheader(
        "Titration, Speciation, and Composition Graphs"
    )

    maximum_hcl = max(
        1.9,
        hcl_mM * 1.2
    )

    hcl_values = np.linspace(
        0,
        maximum_hcl,
        100
    )

    curve = [
        calculate_equilibrium(
            naoh,
            hcl_value,
            families,
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
        [item["pH"] for item in curve],
        color="#1769aa",
        linewidth=2
    )

    axes[0].scatter(
        [hcl_mM],
        [result["pH"]],
        color="black",
        zorder=5,
        label="Current input"
    )

    axes[0].set_xlabel("HCl addition (mM)")
    axes[0].set_ylabel("pH")
    axes[0].set_ylim(0, 14)
    axes[0].set_title("Titration Curve")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    graph_species = [
        "H+",
        "OH-"
    ]

    for family in families:
        graph_species.extend(family["names"])

    for species_name in graph_species:
        axes[1].plot(
            hcl_values,
            [
                item["species"][species_name] * 1000
                for item in curve
            ],
            linewidth=2,
            label=species_name
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("HCl addition (mM)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].set_title("All Interacting Species")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3)

    pH_values = np.linspace(
        0,
        14,
        250
    )

    for family in families:
        fractions = np.array([
            family_distribution(
                10 ** (-pH),
                1.0,
                family["pKas"]
            )
            for pH in pH_values
        ])

        for index, species_name in enumerate(
            family["names"]
        ):
            axes[2].plot(
                pH_values,
                fractions[:, index] * 100,
                linewidth=2,
                label=species_name
            )

    axes[2].set_xlabel("pH")
    axes[2].set_ylabel("Fraction of family (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Species Distribution Versus pH")
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=3)

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
        "Download Graphs as PNG",
        graph_buffer.getvalue(),
        "AquaEquil_graphs.png",
        "image/png"
    )

    st.markdown(
        "### Family Composition During Titration"
    )

    family_names = [
        family["label"]
        for family in families
    ]

    family_totals = np.array([
        [
            sum(
                item["species"][name]
                for name in family["names"]
            ) * 1000
            for family in families
        ]
        for item in curve
    ]).T

    fig2, ax2 = plt.subplots(
        figsize=(11, 5)
    )

    ax2.stackplot(
        hcl_values,
        family_totals,
        labels=family_names,
        alpha=0.85
    )

    ax2.set_xlabel("HCl addition (mM)")
    ax2.set_ylabel("Family concentration (mM)")
    ax2.set_title(
        "Total Concentration of Each Selected Acid Family"
    )
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    fig2.tight_layout()

    st.pyplot(fig2)

    graph_buffer_2 = io.BytesIO()

    fig2.savefig(
        graph_buffer_2,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    st.download_button(
        "Download Composition Graph as PNG",
        graph_buffer_2.getvalue(),
        "AquaEquil_composition.png",
        "image/png"
    )


with about_tab:
    st.subheader("About the Model")

    st.markdown(
        """
The model solves electroneutrality for H+ using Brent's root finder.

Every selected family is represented by all of its protonation states.

H+ and OH- are always included in the result table and speciation graph.

Family concentrations are interpreted as sodium salts of the fully
deprotonated form.

The model assumes ideal aqueous behavior, constant volume, 25°C pKa
values, and no precipitation, complexation, redox, or activity corrections.
"""
    )
