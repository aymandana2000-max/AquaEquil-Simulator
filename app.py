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
