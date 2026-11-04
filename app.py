

import csv
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ============================================================
# EQUILIBRIUM MODEL
# ============================================================

def calculate_equilibrium(
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
    ph = -np.log10(h)

    charge_residual = h + sodium - oh - ocl - chloride
    mass_error = hocl + ocl - chlorine_total

    return {
        "H": h,
        "OH": oh,
        "HOCl": hocl,
        "OCl": ocl,
        "pH": ph,
        "charge_residual": charge_residual,
        "mass_error": mass_error
    }


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AquaEquil Simulator",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 AquaEquil: Chemical Equilibrium Simulator")
st.caption(
    "NaOH–NaOCl–HCl acid–base equilibrium, validation, and speciation"
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        overflow: hidden;
    }

    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Chemical Inputs")

naoh = st.sidebar.number_input(
    "NaOH concentration (M)",
    min_value=0.0,
    value=0.001,
    format="%.8f"
)

naocl = st.sidebar.number_input(
    "NaOCl concentration (M)",
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

pka = st.sidebar.number_input(
    "HOCl pKa",
    min_value=0.0,
    value=7.53,
    format="%.4f"
)

kw = st.sidebar.number_input(
    "Water Kw",
    min_value=0.0,
    value=1e-14,
    format="%.2e"
)

calculate_button = st.sidebar.button(
    "Calculate Equilibrium",
    type="primary"
)


# ============================================================
# CURRENT EQUILIBRIUM RESULT
# ============================================================

result = calculate_equilibrium(
    naoh,
    naocl,
    hcl_mM,
    pka,
    kw
)

if result["HOCl"] > result["OCl"]:
    dominant_species = "HOCl"
else:
    dominant_species = "OCl-"


# ============================================================
# TABS
# ============================================================

calculator_tab, validation_tab, graphs_tab, about_tab = st.tabs(
    [
        "Equilibrium Calculator",
        "Validation",
        "Graphs",
        "About the Model"
    ]
)


# ============================================================
# CALCULATOR TAB
# ============================================================

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
            f"{hcl_mM:.2f} mM"
        )

    st.divider()

    result_col, explanation_col = st.columns(2)

    with result_col:

        st.markdown("### Equilibrium Concentrations")

        result_table = pd.DataFrame({
            "Species": ["H+", "OH-", "HOCl", "OCl-"],
            "Concentration (M)": [
                result["H"],
                result["OH"],
                result["HOCl"],
                result["OCl"]
            ]
        })

        result_table["Concentration (M)"] = (
            result_table["Concentration (M)"].map(
                lambda x: f"{x:.6e}"
            )
        )

        st.table(result_table)

        st.markdown("### Balance Checks")

        st.write(
            f"Charge residual: "
            f"`{result['charge_residual']:.6e} M`"
        )

        st.write(
            f"Mass error: "
            f"`{result['mass_error']:.6e} M`"
        )

        if (
            abs(result["charge_residual"]) < 1e-7
            and abs(result["mass_error"]) < 1e-7
        ):
            st.success("Mass and charge balances are satisfied.")
        else:
            st.warning("Check the balance errors.")

    with explanation_col:

        st.markdown("### Interpretation")

        st.info(
            f"The calculated pH is {result['pH']:.3f}. "
            f"Because pH is "
            f"{'greater' if result['pH'] > pka else 'less'} "
            f"than the pKa ({pka:.2f}), the dominant species is "
            f"**{dominant_species}**."
        )

        st.markdown("### Current Conditions")

        conditions = pd.DataFrame({
            "Parameter": [
                "NaOH",
                "NaOCl",
                "HCl",
                "HOCl pKa",
                "Kw"
            ],
            "Value": [
                f"{naoh:.6e} M",
                f"{naocl:.6e} M",
                f"{hcl_mM:.4f} mM",
                f"{pka:.4f}",
                f"{kw:.2e}"
            ]
        })

        st.table(conditions)


# ============================================================
# VALIDATION TAB
# ============================================================

with validation_tab:

    st.subheader("Five-Case Model Validation")

    hcl_cases = [0.00, 0.50, 1.00, 1.50, 1.90]

    fixed_reference_pH = {
        0.00: 11.00,
        0.50: 10.70,
        1.00: 9.26,
        1.50: 7.53,
        1.90: 6.58
    }

    default_case = (
        np.isclose(naoh, 0.001)
        and np.isclose(naocl, 0.001)
        and np.isclose(pka, 7.53)
        and np.isclose(kw, 1e-14)
    )

    validation_rows = []

    for hcl in hcl_cases:

        current = calculate_equilibrium(
            naoh,
            naocl,
            hcl,
            pka,
            kw
        )

        calculated_pH = current["pH"]
        charge_error = abs(current["charge_residual"])
        mass_error = abs(current["mass_error"])

        if default_case:

            expected_pH = fixed_reference_pH[hcl]
            difference = calculated_pH - expected_pH

            passed = (
                abs(difference) <= 0.10
                and charge_error < 1e-7
                and mass_error < 1e-7
            )

            status = "PASS" if passed else "CHECK"

        else:

            expected_pH = np.nan
            difference = np.nan

            balanced = (
                charge_error < 1e-7
                and mass_error < 1e-7
            )

            status = "BALANCE OK" if balanced else "CHECK"

        validation_rows.append({
            "HCl (mM)": hcl,
            "Calculated pH": calculated_pH,
            "Expected pH": expected_pH,
            "Difference": difference,
            "Charge residual (M)": current["charge_residual"],
            "Mass error (M)": current["mass_error"],
            "Status": status
        })

    validation_df = pd.DataFrame(validation_rows)

    if default_case:
        st.success(
            "Default case: calculated pH is compared with independent "
            "assignment reference values."
        )
    else:
        st.info(
            "Custom case: only mass and charge balance are checked. "
            "Independent reference pH values do not apply."
        )

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

    csv_data = validation_df.to_csv(index=False)

    st.download_button(
        label="Download Validation Results as CSV",
        data=csv_data,
        file_name="AquaEquil_validation_results.csv",
        mime="text/csv"
    )


# ============================================================
# GRAPHS TAB
# ============================================================

with graphs_tab:

    st.subheader("Titration and Speciation Graphs")

    hcl_values = np.linspace(0, 1.9, 100)

    pH_values = []
    hocl_values = []
    ocl_values = []

    for hcl in hcl_values:

        current = calculate_equilibrium(
            naoh,
            naocl,
            hcl,
            pka,
            kw
        )

        pH_values.append(current["pH"])
        hocl_values.append(current["HOCl"] * 1000)
        ocl_values.append(current["OCl"] * 1000)

    marker_hcl = hcl_cases
    marker_pH = []

    for hcl in marker_hcl:

        marker_result = calculate_equilibrium(
            naoh,
            naocl,
            hcl,
            pka,
            kw
        )

        marker_pH.append(marker_result["pH"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8)
    )

    axes[0].plot(
        hcl_values,
        pH_values,
        color="#1769aa",
        linewidth=2,
        label="Calculated pH"
    )

    axes[0].scatter(
        marker_hcl,
        marker_pH,
        color="red",
        marker="x",
        s=70,
        linewidths=2,
        label="Validation points"
    )

    axes[0].scatter(
        [hcl_mM],
        [result["pH"]],
        color="black",
        marker="o",
        s=90,
        zorder=5,
        label="Current input"
    )

    axes[0].set_title(
        f"Titration Curve: NaOH = {naoh:g} M, "
        f"NaOCl = {naocl:g} M"
    )
    axes[0].set_xlabel("HCl addition (mM)")
    axes[0].set_ylabel("pH")
    axes[0].set_xlim(0, 1.9)
    axes[0].set_ylim(0, 14)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        hcl_values,
        hocl_values,
        color="#d62728",
        linewidth=2,
        label="HOCl"
    )

    axes[1].plot(
        hcl_values,
        ocl_values,
        color="#2ca02c",
        linewidth=2,
        label="OCl-"
    )

    axes[1].set_title("Hypochlorous Acid Speciation")
    axes[1].set_xlabel("HCl addition (mM)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].set_xlim(0, 1.9)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

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
        label="Download Graphs as PNG",
        data=graph_buffer.getvalue(),
        file_name="AquaEquil_graphs.png",
        mime="image/png"
    )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.subheader("About the Model")

    st.markdown(
        """
        ### Chemical system

        The simulator models the aqueous NaOH–NaOCl–HCl system at 25°C.

        ### Equilibrium reactions

        **Hypochlorous acid dissociation**

        HOCl ⇌ H⁺ + OCl⁻

        **Water dissociation**

        H₂O ⇌ H⁺ + OH⁻

        ### Mass balance

        [HOCl] + [OCl⁻] = total chlorine concentration

        ### Charge balance

        [H⁺] + [Na⁺] =
        [OH⁻] + [OCl⁻] + [Cl⁻]

        ### Numerical method

        Brent's root-finding method is used to find the hydrogen-ion
        concentration that satisfies the charge balance.

        ### Model assumptions

        - Ideal aqueous solution
        - Constant temperature of 25°C
        - Constant solution volume
        - No chlorine decomposition
        - No redox reactions
        - Only the listed acid–base species are considered

        This is a computational educational model.
        Hypochlorite solutions should not be acidified experimentally.
        """
    )
