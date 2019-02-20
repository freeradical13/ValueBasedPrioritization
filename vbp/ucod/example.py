import io
import pandas
import matplotlib
import statsmodels
import matplotlib.pyplot
import statsmodels.tsa.api

cause = "Malignant neoplasms"
csv_data = """Year,CrudeRate
1999,197.0
2000,196.5
2001,194.3
2002,193.7
2003,192.0
2004,189.2
2005,189.3
2006,187.6
2007,186.9
2008,186.0
2009,185.0
2010,186.2
2011,185.1
2012,185.6
2013,185.0
2014,185.6
2015,185.4
2016,185.1
2017,183.9
"""

def ets_non_seasonal(df, color, predict, exponential=False, damped=False, damping_slope=0.98):
  fit = statsmodels.tsa.api.Holt(df, exponential=exponential, damped=damped).fit(damping_slope=damping_slope if damped else None)
  fit.fittedvalues.plot(color=color, style="--", label='_nolegend_')
  title = "ETS(A,{}{},N)".format("M" if exponential else "A", "_d" if damped else "")
  forecast = fit.forecast(predict).rename("${}$".format(title))
  forecast.plot(color=color, legend=True, style="--")
  print("{}={}".format(title, fit.aicc))

df = pandas.read_csv(io.StringIO(csv_data), index_col="Year", parse_dates=True)
df.plot(color="black", marker="o", legend=True)
ets_non_seasonal(df, "red", 5, exponential=False, damped=False, damping_slope=0.98)
ets_non_seasonal(df, "cyan", 5, exponential=False, damped=True, damping_slope=0.98)
ets_non_seasonal(df, "green", 5, exponential=True, damped=False, damping_slope=0.98)
ets_non_seasonal(df, "blue", 5, exponential=True, damped=True, damping_slope=0.98)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
