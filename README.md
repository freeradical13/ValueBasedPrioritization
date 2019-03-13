# ValueBasedPrioritization

## Article

* PDF: [value_based_prioritization.pdf](value_based_prioritization.pdf)
* TeX Source: [value_based_prioritization.tex](value_based_prioritization.tex)

## Source Code

Prerequisites:

    pip3 install numpy pandas matplotlib statsmodels scipy fbprophet

### Usage

    python3 -m vbp.run -h

## Running

The model type is specified with `--ets`, `--ols`, and/or `--prophet`.
These are not mutually exclusive; if combined, an average is taken of
the results. The default is `--ets`.

By default, action names are obfuscated to reduce bias during model
building and testing. Specify `--do-not-obfuscate` to show actual names.

Some data sources have different data types (e.g. mutually exclusive
groupings of data). Add the `-a` argument before the data source
name to run for all data types. Add the `--data-type X` argument after
the data source name to specify a specific data type.

In general, a list of actions may be specified to run for just that
list; otherwise, without such a list, all actions are processed.

Examples:

    python3 -m vbp.run modeled_value_based_prioritization UnderlyingCausesOfDeathUnitedStates

    python3 -m vbp.run modeled_value_based_prioritization UnderlyingCausesOfDeathUnitedStates --do-not-obfuscate "Ischemic heart diseases" Malaria

### Exponential Smoothing

Using [exponential smoothing](https://otexts.com/fpp2/expsmooth.html):

    python3 -m vbp.run modeled_value_based_prioritization ${DATA_SOURCE} --ets

Specify `--ets-no-multiplicative-models` to only use additive models.

Specify `--ets-no-additive-models` to only use multiplicative models.

### Linear Regression

Using [linear regression](https://otexts.com/fpp2/regression.html).

    python3 -m vbp.run modeled_value_based_prioritization ${DATA_SOURCE} --ols

Specify `--ols-max-degrees X` to model higher degrees.

### Prophet

Using [Facebook Prophet](https://facebook.github.io/prophet/).

    python3 -m vbp.run modeled_value_based_prioritization ${DATA_SOURCE} --prophet

### United States

As of 2019-03-01, the unzipped U.S. mortality data consumes ~36GB of
disk. It will be downloaded and unzipped automatically when a function
is used that needs it.

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

Final output:

    python3 -m vbp.run modeled_value_based_prioritization UnderlyingCausesOfDeathUnitedStates --data-type UCOD_LONGTERM_COMPARABLE_LEADING

### World

As of 2019-03-01, the unzipped World mortality data consumes ~320MB of
disk. It will be downloaded and unzipped automatically when a function
is used that needs it.
