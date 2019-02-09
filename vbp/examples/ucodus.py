import sys
import math
import numpy
import pandas
import argparse
import datetime
import matplotlib
import matplotlib.offsetbox
import statsmodels.tools
import statsmodels.formula.api

def load_codus(file):
  df = pandas.read_csv(
         file,
         sep="\t",
         usecols=["Year", "ICD Sub-Chapter", "ICD Sub-Chapter Code", "Deaths", "Population", "Crude Rate"],
         na_values=["Unreliable"],
         parse_dates=[0]
       ).dropna(how="all")
  df.rename(columns = {"Year": "Date"}, inplace=True)
  df["Year"] = df["Date"].dt.year
  return df

def model_linear_regression(df, x, y, degree=1):
  data = {"x": df[x].values, "y": df[y].values}
  formula = "y ~ " + " + ".join("I(x**{})".format(i) for i in range(1, degree+1))
  print("Formula: {}".format(formula))
  return statsmodels.formula.api.ols(formula, data).fit()

def create_plot(df, cause, predict, degree=1):
  df = df[df["ICD Sub-Chapter"] == cause]
  max_year = df.Year.values[-1]
  
  min_scaled_domain = -1
  max_scaled_domain = +1
  
  current_year = datetime.datetime.now().year
  year_diff = current_year - max_year
  if year_diff < 0:
    start = max_year + 1
    end = start + predict
  else:
    start = max_year + 1
    end = current_year + predict
  
  nan_row = {i: numpy.nan for i in df.columns.get_values().tolist()}
  for i in range(start, end):
    new_row = nan_row.copy()
    new_row["Year"] = i
    df = df.append(new_row, ignore_index=True)
  
  df["ScaledYear"] = df["Year"].transform(lambda x: numpy.interp(x, (x.min(), x.max()), (min_scaled_domain, max_scaled_domain)))
  
  print(df)

  model = model_linear_regression(df, "ScaledYear", "Crude Rate", degree)
  print(model.summary())

  ax = df.plot("Year", "Crude Rate", kind="scatter", grid=True, title="Deaths from {}".format(cause), color = 'red')

  func = numpy.polynomial.Polynomial(model.params)
  matplotlib.pyplot.plot(df["Year"], func(df["ScaledYear"]), color="blue")
  
  end -= 1
  predicted = func(max_scaled_domain)
  print("Predicted in {} years ({}): {}".format(predict, end, predicted))
  matplotlib.pyplot.plot([end], [predicted], "cD") # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
  
  ax.add_artist(matplotlib.offsetbox.AnchoredText("$x^{}$; $\\barR^2$ = {:0.3f}".format(degree, model.rsquared_adj), loc="upper center"))
  ax.add_artist(matplotlib.offsetbox.AnchoredText("Predicted in +{} = {:0.1f}".format(predict, predicted), loc="upper right", pad=0.65))

  fig = matplotlib.pyplot.gcf()
  fig.autofmt_xdate(bottom=0.2, rotation=30, ha="right", which="both")
  ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
  #fig.set_size_inches(10, 5)
  matplotlib.pyplot.tight_layout()
  cleaned_title = cause.replace(" ", "_").replace("(", "").replace(")", "")
  matplotlib.pyplot.savefig("{}_{}.png".format(cleaned_title, degree), dpi=100)
  df.to_csv("{}.csv".format(cleaned_title))
  matplotlib.pyplot.show()

def run_regressions(args):
  parser = argparse.ArgumentParser()
  parser.add_argument("cause", help="ICD Sub-Chapter")
  parser.add_argument("-f", "--file", help="path to file", default="Underlying Cause of Death, 1999-2017.txt")
  parser.add_argument("-m", "--min-degrees", help="minimum polynomial degree", type=int, default=1)
  parser.add_argument("-x", "--max-degrees", help="maximum polynomial degree", type=int, default=4)
  parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=5)
  options = parser.parse_args(args)
  
  df = load_codus(options.file)
  for i in range(options.min_degrees, options.max_degrees + 1):
    create_plot(df, options.cause, options.predict, i)

if __name__ == "__main__":
  run_regressions(sys.argv[1:])
