# ValueBasedPrioritization

## Article

* PDF: [value_based_prioritization.pdf](value_based_prioritization.pdf)
* LaTeX Source: [value_based_prioritization.tex](value_based_prioritization.tex)

## Source Code

Prerequisites:

    pip3 install numpy pandas matplotlib statsmodels scipy

### Usage

    python3 -m vbp.run -h

### Underlying causes of death in the United States

Example:

    python3 -m vbp.run predict UnderlyingCausesOfDeathUnitedStates "Malignant neoplasms"

## Running

### United States

#### Long-term, comparable, leading causes of death

Generate `data/ucod/united_states/comparable_data_since_1959.xlsx` for
all long-term, comparable, leading causes of death in
https://www.cdc.gov/nchs/data/dvs/lead1900_98.pdf:

    python3 -m vbp.run prepare_data UnderlyingCausesOfDeathUnitedStates

Rows 1900:1957 and the sheet `Comparability Ratios` in
`data/ucod/united_states/comparable_ucod_estimates.xlsx` were manually
input from https://www.cdc.gov/nchs/data/dvs/lead1900_98.pdf

Open `comparable_data_since_1959.xlsx` and copy rows 1959:Present.

Open `comparable_ucod_estimates.xlsx` and paste on top starting
at 1959.

Process `comparable_ucod_estimates.xlsx` with its
`Comparability Ratios` sheet to generate
`comparable_ucod_estimates_ratios_applied.xlsx`:

    python3 -m vbp.run prepare_data UnderlyingCausesOfDeathUnitedStates --comparable-ratios
