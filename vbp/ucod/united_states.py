import sys
import vbp
import math
import numpy
import scipy
import pandas
import argparse
import datetime
import matplotlib
import matplotlib.pyplot
import matplotlib.offsetbox
import statsmodels.tools
import statsmodels.stats.api
import statsmodels.stats.diagnostic
import statsmodels.formula.api

class UnderlyingCausesOfDeathUnitedStates(vbp.DataSource):
  def initialize_parser(self, parser):
    parser.add_argument("cause", nargs="*", help="ICD Sub-Chapter")
    parser.add_argument("-f", "--file", help="path to file", default="data/ucod/united_states/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
    parser.add_argument("-m", "--min-degrees", help="minimum polynomial degree", type=int, default=1)
    parser.add_argument("-x", "--max-degrees", help="maximum polynomial degree", type=int, default=4)
    parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose", default=False)
    parser.add_argument("-d", "--hide-graphs", dest="show_graphs", action="store_false", help="hide graphs")
    parser.add_argument("-s", "--show-graphs", dest="show_graphs", action="store_true", help="verbose")
    parser.set_defaults(
      show_graphs=False,
    )

  def run_load(self):
    df = pandas.read_csv(
          self.options.file,
          sep="\t",
          usecols=["Year", "ICD Sub-Chapter", "ICD Sub-Chapter Code", "Deaths", "Population"],
          na_values=["Unreliable"],
          parse_dates=[0]
        ).dropna(how="all")
    df.rename(columns = {"Year": "Date"}, inplace=True)
    df["Year"] = df["Date"].dt.year
    # https://wonder.cdc.gov/wonder/help/cmf.html#Frequently%20Asked%20Questions%20about%20Death%20Rates
    df["Crude Rate"] = (df.Deaths / df.Population) * 100000.0
    return df

  def get_action_column_name(self):
    return "ICD Sub-Chapter"
  
  def run_predict(self):
    r_data = {
      "AdjustedR2": {},
      "ProbFStatistic": {},
      "LogLikelihood": {},
      "AIC": {},
      "BIC": {},
      "BreuschPagan": {},
      "Formula": {},
    }
    
    causes = self.options.cause
    if causes is None or len(causes) == 0:
      causes = self.get_possible_actions()
    
    for cause in causes:
      try:
        for i in range(self.options.min_degrees, self.options.max_degrees + 1):
          self.create_plot(cause, i, self.options.predict, r_data)
      except:
        e = sys.exc_info()[0]
        print("Error {} processing {}".format(e, cause))
        break
    
    r = pandas.DataFrame(r_data)
    r.index.rename(["Action", "Degree"], inplace=True)
    vbp.print_full_columns(r)
    r.to_csv(self.create_output_name("r.csv"))

  def create_plot(self, action, degree, predict, r_data):
    df = self.run_get_action_data(action)
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
    
    df["ScaledYear"] = df["Year"].transform(vbp.normalize0to1)
    
    dfnoNaNs = df.dropna()
    if self.options.verbose:
      print(dfnoNaNs)

    formula = vbp.linear_regression_formula(degree)
    model = vbp.linear_regression(dfnoNaNs, "ScaledYear", "Crude Rate", formula)
    if self.options.verbose:
      print(model.summary())
    
    r_data["AdjustedR2"][(self.get_obfuscated_name(action), degree)] = model.rsquared_adj
    r_data["ProbFStatistic"][(self.get_obfuscated_name(action), degree)] = model.f_pvalue
    r_data["LogLikelihood"][(self.get_obfuscated_name(action), degree)] = model.llf
    r_data["AIC"][(self.get_obfuscated_name(action), degree)] = model.aic
    r_data["BIC"][(self.get_obfuscated_name(action), degree)] = model.bic
    r_data["Formula"][(self.get_obfuscated_name(action), degree)] = vbp.linear_regression_modeled_formula(model, degree)
    
    #predicted = model.fittedvalues.copy()
    #actual = dfnoNaNs["Crude Rate"].values.copy()
    #residuals = actual - predicted
    
    # Linearity hypothesis test
    # http://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.linear_harvey_collier.html
    #tvalue, pvalue = statsmodels.stats.api.linear_harvey_collier(model)
    #if self.options.verbose:
    #  print("tvalue={}, pvalue={}".format(tvalue, pvalue))
    
    # Residuals plot
    #fig, ax = matplotlib.pyplot.subplots()
    #ax.scatter(predicted, residuals)
    #matplotlib.pyplot.show()
    
    # Normal probability plot
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
    #fig, ax = matplotlib.pyplot.subplots()
    #(osm, osr), (slope, intercept, r) = scipy.stats.probplot(residuals, plot=ax, fit=True)
    #if self.options.verbose:
    #  print(r*r)
    #matplotlib.pyplot.show()
    
    # Breusch-Pagan Lagrange Multiplier test for heteroscedasticity
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
    lm, lm_pvalue, fvalue, f_pvalue = statsmodels.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    if self.options.verbose:
      print("het_breuschpagan: {} {} {} {}".format(lm, lm_pvalue, fvalue, f_pvalue))
    r_data["BreuschPagan"][(self.get_obfuscated_name(action), degree)] = f_pvalue

    ax = df.plot("Year", "Crude Rate", kind="scatter", grid=True, title="Deaths from {}".format(self.get_obfuscated_name(action)), color = "red")

    func = numpy.polynomial.Polynomial(model.params)
    matplotlib.pyplot.plot(df["Year"], func(df["ScaledYear"]), color="blue")
    
    end -= 1
    predicted = func(max_scaled_domain)
    if self.options.verbose:
      print("Predicted in {} years ({}): {}".format(predict, end, predicted))
    matplotlib.pyplot.plot([end], [predicted], "cD") # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    ax.add_artist(matplotlib.offsetbox.AnchoredText("$x^{}$; $\\barR^2$ = {:0.3f}".format(degree, model.rsquared_adj), loc="upper center"))
    ax.add_artist(matplotlib.offsetbox.AnchoredText("Predicted in +{} = {:0.1f}".format(predict, predicted), loc="upper right", pad=0.65))

    fig = matplotlib.pyplot.gcf()
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha="right", which="both")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    #fig.set_size_inches(10, 5)
    matplotlib.pyplot.tight_layout()
    cleaned_title = self.get_obfuscated_name(action).replace(" ", "_").replace("(", "").replace(")", "")
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}.png".format(cleaned_title, degree)), dpi=100)
    df.to_csv(self.create_output_name("{}.csv").format(cleaned_title))
    
    if self.options.show_graphs:
      matplotlib.pyplot.show()
      
    matplotlib.pyplot.close(fig)
