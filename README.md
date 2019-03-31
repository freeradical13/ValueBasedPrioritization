# ValueBasedPrioritization

## Article

Academic article: https://github.com/freeradical13/ValueBasedPrioritization/raw/master/value_based_prioritization.pdf

## Installation

https://pypi.org/project/vbp/

    pip3 install vbp

## vbp

Value Based Prioritization (`vbp`) uses value theory to quantitatively
prioritize potential actions to accomplish a goal.

The `vbp.run` module may be used from the command line to perform
different VBP actions such as listing actions (`list`), counting
actions (`count`), predicting values (`predict`), running Modeled VBP
(`modeled_value_based_prioritization`), and more. For usage, run:

    python3 -m vbp.run

Any non-terminal output goes to the `vbpoutput` sub-folder.

Here is a simple example counting the number of groupings of
underlying causes of death for the United States for the default data
type:

    python3 -m vbp.run count UCODUnitedStates

Alternatively, if installed through `pip`, a `vbp` script wrapper may
be used:

    vbp count UCODUnitedStates

The academic article above includes footnotes with details on how to
run `vbp` to produce the output of each step.

This package provides abstract classes and utility methods to run
VBP, mostly focused on Modeled VBP which uses time series data to
predict future values and prioritize actions based on the relative
predicted values.

The `vbp.DataSource` class is the base abstract class for VBP.

The `vbp.TimeSeriesDataSource` abstract class inherits from
`vbp.DataSource` and may be used for Modeled VBP. The
`vbp.ExampleDataSource` class demonstrates a simple data source based
on `vbp.TimeSeriesDataSource`.

Built-in Modeled VBPs include Underlying Cause of Death models for
the United States (`vbp.ucod.united_states.UCODUnitedStates`) and the
World (`vbp.ucod.world.UCODWorld`). These
data sources both inherit from `vbp.ucod.icd.ICDDataSource` which
inherits from `vbp.TimeSeriesDataSource`.

## Running

The model type is specified with `--ets`, `--ols`, and/or `--prophet`.
These are not mutually exclusive; if combined during
`modeled_value_based_prioritization`, an average is taken of the
results. The default is `--ets`.

By default, action names are obfuscated to reduce bias during model
building and testing. Specify `--do-not-obfuscate` to show actual names.

Some data sources have different data types (e.g. mutually exclusive
groupings of data). Add the `-a` argument before the data source
name to run for all data types. Add the `--data-type X` argument after
the data source name to specify a specific data type.

In general, a list of actions may be specified to run for just that
list; otherwise, without such a list, all actions are processed.

Examples:

    python3 -m vbp.run modeled_value_based_prioritization UCODUnitedStates

    python3 -m vbp.run modeled_value_based_prioritization UCODUnitedStates --do-not-obfuscate "Ischemic heart diseases" Malaria

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

    python3 -m vbp.run prepare_data UCODUnitedStates

Rows 1900:1957 and the sheet `Comparability Ratios` in
`data/ucod/united_states/comparable_ucod_estimates.xlsx` were manually
input from https://www.cdc.gov/nchs/data/dvs/lead1900_98.pdf

Open `comparable_data_since_1959.xlsx` and copy rows 1959:Present.

Open `comparable_ucod_estimates.xlsx` and paste on top starting
at 1959.

Process `comparable_ucod_estimates.xlsx` with its
`Comparability Ratios` sheet to generate
`comparable_ucod_estimates_ratios_applied.xlsx`:

    python3 -m vbp.run prepare_data UCODUnitedStates --comparable-ratios

Final output:

    python3 -m vbp.run modeled_value_based_prioritization UCODUnitedStates --data-type UCOD_LONGTERM_COMPARABLE_LEADING

### World

As of 2019-03-01, the unzipped World mortality data consumes ~320MB of
disk. It will be downloaded and unzipped automatically when a function
is used that needs it.

When testing, writing data spreadsheets takes a lot of time and may
be avoided with --do-not-write-spreadsheets.

### Creating a new Data Source

Review vbp/example.py for a simple example. The basic process is:

1. Create a sub-class of vbp.DataSource in somename.py
1. Implement all `@abc.abstractmethod` methods and override any
   other superclass methods as needed.
1. Import somename.py at the top of vbp/run.py

## Development

Prerequisites:

    pip3 install numpy pandas matplotlib statsmodels scipy fbprophet

Updating PyPI package:

    # Edit version in setup.py and __init__.py
    python3 setup.py bdist bdist_wheel
    python3 -m twine upload --skip-existing dist/*
    # https://pypi.org/project/vbp/
