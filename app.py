import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq


st.set_page_config(page_title="AquaEquil Simulator", layout="wide")


# -----------------------------------------------------------------------------
# General acid-family equilibrium model
# -----------------------------------------------------------------------------
def family_distribution(h, total, pKas):
    """Return concentrations of every protonation state of a family.

    pKas are listed from the most protonated state to the least protonated
    state.  For example, carbonate uses H2CO3/HCO3-/CO3-- and two pKas.
    """
    kas = 10.0 ** (-np.asarray(pKas, dtype=float))
    n = len(kas)
    terms = [h ** n]
    for i in range(1, n + 1):
        terms.append(np.prod(kas[:i]) * h ** (n - i))
    fractions = np.asarray(terms) / np.sum(terms)
    return total * fractions


def calculate_equilibrium(naoh, hcl_mM, families, kw=1e-14):
    hcl = hcl_mM / 1000.0

    # Each family input is treated as a sodium salt of its most deprotonated
    # form, so it contributes charge_max sodium equivalents per mole.
    sodium = naoh + sum(f["total"] * len(f["pKas"]) for f in families)
    chloride = hcl

    def charge_balance(log_h):
        h = 10.0 ** log_h
        oh = kw / h
        negative_species = oh + chloride
        for family in families:
            states = family_distribution(h, family["total"], family["pKas"])
            negative_species += sum(i * states[i] for i in range(len(states)))
        return h + sodium - negative_species

    log_h = brentq(charge_balance, -14, 0)
    h = 10.0 ** log_h
    oh = kw / h

    species = {"H+": h, "OH-": oh}
    for family in families:
        states = family_distribution(h, family["total"], family["pKas"])
        for name, concentration in zip(family["names"], states):
            species[name] = concentration

    charge_residual = charge_balance(log_h)
    mass_errors = {
        family["label"]: sum(species[name] for name in family["names"])
        - family["total"]
        for family in families
    }
    return {
        "H+": h,
        "OH-": oh,
        "pH": -np.log10(h),
        "species": species,
        "charge_residual": charge_residual,
        "mass_errors": mass_errors,
    }


GROUPS = {
    "OCl (hypochlorite)": {
        "names": ["HOCl", "OCl-"], "default_pKas": [7.53],
        "formula": "HOCl ⇌ H+ + OCl-"
    },
    "CO3 (carbonate)": {
        "names": ["H2CO3", "HCO3-", "CO3--"], "default_pKas": [6.35, 10.33],
        "formula": "H2CO3 ⇌ H+ + HCO3- ⇌ 2H+ + CO3--"
    },
    "SO4 (sulfate)": {
        "names": ["H2SO4", "HSO4-", "SO4--"], "default_pKas": [-3.0, 1.99],
        "formula": "H2SO4 ⇌ H+ + HSO4- ⇌ 2H+ + SO4--"
    },
    "PO4 (phosphate)": {
        "names": ["H3PO4", "H2PO4-", "HPO4--", "PO4---"],
        "default_pKas": [2.15, 7.20, 12.35],
        "formula": "H3PO4 ⇌ H+ + H2PO4- ⇌ H+ + HPO4-- ⇌ H+ + PO4---"
    },
}


st.title("AquaEquil: Chemical Equilibrium Simulator")
st.caption("General multi-family acid–base equilibrium, validation, and speciation")
st.sidebar.header("Solution inputs")
naoh = st.sidebar.number_input("NaOH concentration (M)", min_value=0.0, value=0.001, format="%.8f")
hcl_mM = st.sidebar.number_input("HCl concentration (mM)", min_value=0.0, value=0.0, format="%.4f")
kw = st.sidebar.number_input("Water Kw", min_value=0.0, value=1e-14, format="%.2e")

st.sidebar.header("Acid families")
selected = st.sidebar.multiselect(
    "Select species groups",
    list(GROUPS),
    default=["OCl (hypochlorite)"],
)

families = []
for label in selected:
    info = GROUPS[label]
    st.sidebar.markdown(f"**{label}**")
    total = st.sidebar.number_input(
        f"{label} total concentration (M)", min_value=0.0, value=0.001,
        format="%.8f", key=f"total_{label}"
    )
    pKas = []
    for i, default in enumerate(info["default_pKas"]):
        pKas.append(st.sidebar.number_input(
            f"{label} pKa {i + 1}", value=float(default), format="%.4f",
            key=f"pka_{label}_{i}"
        ))
    families.append({
        "label": label, "total": total, "pKas": pKas,
        "names": info["names"], "formula": info["formula"]
    })
if not families:
    st.warning("Select at least one species group.")
    families = [{
        "label": "OCl (hypochlorite)", "total": 0.0, "pKas": [7.53],
        "names": GROUPS["OCl (hypochlorite)"]["names"],
        "formula": GROUPS["OCl (hypochlorite)"]["formula"]
    }]

calculate_button = st.sidebar.button("Calculate Equilibrium", type="primary")

# Streamlit reruns the whole script after every widget interaction.  A normal
# button is only True on the one rerun where it is clicked, so keep the
# calculation state explicitly; otherwise the page would stop at the title.
if "has_calculated" not in st.session_state:
    st.session_state.has_calculated = False
if calculate_button:
    st.session_state.has_calculated = True
if not st.session_state.has_calculated:
    st.info("Showing the default calculation. Change inputs and press Calculate Equilibrium to update it.")

result = calculate_equilibrium(naoh, hcl_mM, families, kw)
all_species = result["species"]
dominant_species = max(all_species, key=all_species.get)

calculator_tab, validation_tab, graphs_tab, about_tab = st.tabs(
    ["Equilibrium Calculator", "Validation", "Graphs", "About the Model"]
)

with calculator_tab:
    st.subheader("Equilibrium Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Equilibrium pH", f"{result['pH']:.5f}")
    c2.metric("Dominant Species", dominant_species)
    c3.metric("HCl Input", f"{hcl_mM:.5f} mM")

    result_table = pd.DataFrame({
        "Species": list(all_species),
        "Concentration (M)": list(all_species.values())
    })
    st.dataframe(result_table.style.format({"Concentration (M)": "{:.6e}"}), use_container_width=True)
    st.markdown(f"Charge residual: `{result['charge_residual']:.6e} M`")
    mass_error = max(abs(x) for x in result["mass_errors"].values())
    st.markdown(f"Maximum family mass error: `{mass_error:.6e} M`")
    if abs(result["charge_residual"]) < 1e-7 and mass_error < 1e-7:
        st.success("Mass and charge balances are satisfied.")
    else:
        st.warning("Check the balance errors.")

with validation_tab:
    st.subheader("Balance Validation")
    validation = [{"Family": k, "Mass error (M)": v} for k, v in result["mass_errors"].items()]
    validation.append({"Family": "Charge balance", "Mass error (M)": result["charge_residual"]})
    st.dataframe(pd.DataFrame(validation).style.format({"Mass error (M)": "{:.6e}"}), use_container_width=True)
    st.download_button("Download Validation Results as CSV", pd.DataFrame(validation).to_csv(index=False),
                       "AquaEquil_validation_results.csv", "text/csv")

with graphs_tab:
    st.subheader("Titration, Speciation, and Composition Graphs")
    hcl_values = np.linspace(0, max(1.9, hcl_mM * 1.2), 100)
    curve = [calculate_equilibrium(naoh, x, families, kw) for x in hcl_values]
    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    axes[0].plot(hcl_values, [x["pH"] for x in curve], color="#1769aa", linewidth=2)
    axes[0].scatter([hcl_mM], [result["pH"]], color="black", zorder=5, label="Current input")
    axes[0].set(xlabel="HCl addition (mM)", ylabel="pH", ylim=(0, 14), title="Titration curve")
    axes[0].grid(alpha=0.3); axes[0].legend()
    for name in ["H+", "OH-"] + [n for f in families for n in f["names"]]:
        axes[1].plot(hcl_values, [x["species"][name] * 1000 for x in curve], linewidth=2, label=name)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="HCl addition (mM)", ylabel="Concentration (mM)", title="All interacting species")
    axes[1].grid(alpha=0.3); axes[1].legend(ncol=3)

  # Plot each family's fractional composition against pH.  This separates
    # acid-base conversion from the changing total concentration during the
    # HCl titration and makes all pKa transitions visible.
    pH_values = np.linspace(0, 14, 250)
    for family in families:
        fractions = np.array([
            family_distribution(10 ** (-pH), 1.0, family["pKas"])
            for pH in pH_values
        ])
        for i, name in enumerate(family["names"]):
            axes[2].plot(pH_values, fractions[:, i] * 100, linewidth=2, label=name)
    axes[2].set(xlabel="pH", ylabel="Fraction of family (%)", ylim=(0, 100),
                title="Species distribution versus pH")
    axes[2].grid(alpha=0.3); axes[2].legend(ncol=3)
    fig.tight_layout()
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download Graphs as PNG", buf.getvalue(), "AquaEquil_graphs.png", "image/png")
st.markdown("### Family composition during titration")
    family_names = [f["label"] for f in families]
    family_totals = np.array([
        [sum(x["species"][name] for name in family["names"]) * 1000 for family in families]
        for x in curve
    ]).T
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.stackplot(hcl_values, family_totals, labels=family_names, alpha=0.85)
    ax2.set(xlabel="HCl addition (mM)", ylabel="Family concentration (mM)",
            title="Total concentration of each selected acid family")
    ax2.grid(alpha=0.3); ax2.legend(loc="upper left")
    fig2.tight_layout()
    st.pyplot(fig2)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download Composition Graph as PNG", buf2.getvalue(),
                       "AquaEquil_composition.png", "image/png")

with about_tab:
    st.subheader("About the Model")
    st.markdown("""
    The model solves electroneutrality for H+ using Brent's root finder. Every
    selected family is represented by all of its protonation states. H+ and OH-
    are always included in the result table and speciation graph.

    Family concentrations are interpreted as sodium salts of the fully
    deprotonated form. This determines the spectator sodium charge. The model
    assumes ideal aqueous behavior, constant volume, 25°C pKa values, and no
    precipitation, complexation, redox, or activity corrections.
    """)
