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
    value=None,
    step=0.1,
    format="%.1f",
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

if run_validation:
    if any(value is None for value in [
        validation_naoh,
        validation_naocl,
        validation_pka,
        validation_kw
    ]):
        st.warning(
            "Enter all validation concentrations and pKa values first."
        )
        st.stop()

    # Existing validation calculations and graphs go here.
else:
    st.info(
        "Enter the validation values and press Run Validation "
        "to display the table and graphs."
    )
