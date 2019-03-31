Value Based Prioritization (VBP) uses value theory to quantitatively
prioritize potential actions to accomplish a goal.

https://github.com/freeradical13/ValueBasedPrioritization

This package provides abstract classes and utility methods to run
VBP, mostly focused on Modeled VBP which uses time series data to
predict future values and prioritize actions based on the relative
predicted values.

The DataSource class is the base abstract class for VBP.

The TimeSeriesDataSource abstract class inherits from DataSource and
may be used for Modeled VBP. The ExampleDataSource class demonstrates
a simple data source based on TimeSeriesDataSource.

Built-in Modeled VBPs include Underlying Cause of Death models for
the United States (UCODUnitedStates) and the World (UCODWorld). These
data sources both inherit from ICDDataSource which inherits from
TimeSeriesDataSource.

The run module may be used from the command line to perform different
VBP actions such as listing actions (list), counting actions (count),
predicting values (predict), running Modeled VBP
(modeled_value_based_prioritization), and more. For usage, run:

python -m vbp.run

