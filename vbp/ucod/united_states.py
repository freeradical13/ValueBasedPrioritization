import sys
import vbp
import math
import numpy
import scipy
import pandas
import pprint
import argparse
import datetime
import traceback
import matplotlib
import matplotlib.pyplot
import statsmodels.tools
import matplotlib.offsetbox
import statsmodels.stats.api
import statsmodels.formula.api
import statsmodels.stats.diagnostic
import statsmodels.tools.eval_measures

class UnderlyingCausesOfDeathUnitedStates(vbp.DataSource):
  def initialize_parser(self, parser):
    parser.add_argument("cause", nargs="*", help="ICD Sub-Chapter")
    parser.add_argument("-f", "--file", help="path to file", default="data/ucod/united_states/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
    parser.add_argument("-m", "--min-degrees", help="minimum polynomial degree", type=int, default=1)
    parser.add_argument("-x", "--max-degrees", help="maximum polynomial degree", type=int, default=4)
    parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose", default=False)
    parser.add_argument("-d", "--hide-graphs", dest="show_graphs", action="store_false", help="hide graphs")
    parser.add_argument("-s", "--show-graphs", dest="show_graphs", action="store_true", help="verbose")
    parser.add_argument("-l", "--model", choices=["lowest_aic", "lowest_aicc", "lowest_bic", "highest_log_likelihood"], help="Best fitting model algorithm", default="lowest_aicc")
    parser.add_argument("--debug", action="store_true", help="Debus", default=False)
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
      "AICc": {},
      "BIC": {},
      "BreuschPagan": {},
      "Formula": {},
      "Predicted": {},
      "PredictedDerivative": {},
    }
    
    causes = self.options.cause
    if causes is None or len(causes) == 0:
      causes = self.get_possible_actions()
      
    count = len(causes)
    if self.options.verbose:
      print("{} actions".format(count))
    
    action_count = 1
    for cause in causes:
      try:
        for i in range(self.options.min_degrees, self.options.max_degrees + 1):
          self.create_model(cause, i, self.options.predict, r_data, action_count, count)
      except:
        traceback.print_exc()
        break
      action_count += 1
    
    if self.options.verbose:
      print("Result actions: {}".format(len(r_data["AdjustedR2"])))
      pprint.pprint(r_data)
    
    r = pandas.DataFrame(r_data)
    r.index.rename(["Action", "Degree"], inplace=True)
    self.write_spreadsheet(r, "r_raw")
    
    if self.options.verbose:
      print("Actions = {}".format(len(r.groupby("Action"))))
    
    if self.options.verbose:
      print("Before running best_fitting_model")
      print(r)
    
    r = r.groupby(["Action"]).apply(lambda x: self.best_fitting_model(x))
    
    if self.options.verbose:
      print("After running best_fitting_model")
      vbp.print_full_columns(r)
    
    r["S1"] = 1.0
    
    # Adjust any out-of-bounds values
    r["S2"] = r.apply(lambda row: 0.1 if row.AdjustedR2 < 0.1 else 1.0, axis=1)
    r["S3"] = r.apply(lambda row: 0.1 if row.ProbFStatistic > 0.05 else 1.0, axis=1)
    r["S4"] = r.apply(lambda row: 0.1 if row.BreuschPagan < 0.05 else 1.0, axis=1)
    
    if len(r) > 1:
      r["S7"] = vbp.normalize(r.PredictedDerivative.values, 0.5, 1.0)
    else:
      r["S7"] = 1.0
    
    r = r[["Predicted", "S1", "S2", "S3", "S4", "S5", "S7"]]
    r.reset_index(level=1, inplace=True) # drop=True to suppress adding Degree to the column
    vbp.print_full_columns(r)
    
    self.write_spreadsheet(r, "r")
    
    if self.options.debug:
      r.to_pickle(self.create_output_name("data.pkl"))
      
  def best_fitting_model(self, df):
    if self.options.verbose:
      print("Incoming:\n{}\n".format(df))
      
    fit_result = getattr(self, "best_fitting_model_{}".format(self.options.model))(df)
    if fit_result is not None and len(fit_result) == 1:
      df = fit_result
      df["S5"] = 1.0
    else:
      df = df[df.index.get_level_values(1) == 1]
      df["S5"] = 0.1
      
    df.reset_index(level=0, drop=True, inplace=True)
    
    if self.options.verbose:
      print("Outgoing:\n{}\n".format(df))
      
    return df

  def best_fitting_model_lowest_aic(self, df):
    result = None

    tmp = df[(df.ProbFStatistic < 0.05) & (df.BreuschPagan > 0.05)]
    if not tmp.empty:
      tmp = tmp[tmp.AIC == tmp.AIC.min()]
      if not tmp.empty:
        result = tmp
    
    if result is None:
      tmp = df[df.AIC == df.AIC.min()]
      if not tmp.empty:
        result = tmp
        
    return result
  
  def best_fitting_model_lowest_aicc(self, df):
    result = None

    tmp = df[(df.ProbFStatistic < 0.05) & (df.BreuschPagan > 0.05)]
    if not tmp.empty:
      tmp = tmp[tmp.AICc == tmp.AICc.min()]
      if not tmp.empty:
        result = tmp
    
    if result is None:
      tmp = df[df.AICc == df.AICc.min()]
      if not tmp.empty:
        result = tmp
        
    return result
  
  def best_fitting_model_lowest_bic(self, df):
    result = None

    tmp = df[(df.ProbFStatistic < 0.05) & (df.BreuschPagan > 0.05)]
    if not tmp.empty:
      tmp = tmp[tmp.BIC == tmp.BIC.min()]
      if not tmp.empty:
        result = tmp
    
    if result is None:
      tmp = df[df.BIC == df.BIC.min()]
      if not tmp.empty:
        result = tmp
        
    return result
  
  def best_fitting_model_highest_log_likelihood(self, df):
    result = None

    tmp = df[(df.ProbFStatistic < 0.05) & (df.BreuschPagan > 0.05)]
    if not tmp.empty:
      result = tmp[tmp.LogLikelihood == tmp.LogLikelihood.max()]
    
    if result is None:
      result = df[df.LogLikelihood == df.LogLikelihood.max()]
        
    return result
    
  def create_model(self, action, degree, predict, r_data, i, count):
    df = self.run_get_action_data(action)
    df = df.reset_index()
    
    print("Creating model {} of {} for: {}, degrees: {}".format(i, count, self.get_obfuscated_name(action), degree))
      
    max_year = df.Year.values[-1]
    
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
    
    df["ScaledYear"] = numpy.vectorize(vbp.normalize_func_explicit_minmax(df.Year.values, df.Year.min(), max_year, 0, 1))(df.Year.values)
    
    dfnoNaNs = df.dropna()
    if self.options.verbose:
      print(dfnoNaNs)

    formula = vbp.linear_regression_formula(degree)
    model = vbp.linear_regression(dfnoNaNs, "ScaledYear", "Crude Rate", formula)
    self.write_verbose("{}_{}.txt".format(self.get_obfuscated_name(action), degree), model.summary())
    
    r_data["AdjustedR2"][(self.get_obfuscated_name(action), degree)] = model.rsquared_adj
    r_data["ProbFStatistic"][(self.get_obfuscated_name(action), degree)] = model.f_pvalue
    r_data["LogLikelihood"][(self.get_obfuscated_name(action), degree)] = model.llf
    r_data["AIC"][(self.get_obfuscated_name(action), degree)] = model.aic
    r_data["BIC"][(self.get_obfuscated_name(action), degree)] = model.bic
    r_data["Formula"][(self.get_obfuscated_name(action), degree)] = vbp.linear_regression_modeled_formula(model, degree)
    
    r_data["AICc"][(self.get_obfuscated_name(action), degree)] = statsmodels.tools.eval_measures.aicc(model.llf, model.nobs, degree + 1)
    
    predicted_values = model.fittedvalues.copy()
    actual = dfnoNaNs["Crude Rate"].values.copy()
    residuals = actual - predicted_values
    
    # Linearity hypothesis test
    # http://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.linear_harvey_collier.html
    #tvalue, pvalue = statsmodels.stats.api.linear_harvey_collier(model)
    #if self.options.verbose:
    #  print("tvalue={}, pvalue={}".format(tvalue, pvalue))
    
    # Residuals plot
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(predicted_values, residuals)
    ax.set_title("Residuals of {} ($x^{}$)".format(self.get_obfuscated_name(action), degree))
    matplotlib.pyplot.tight_layout()
    #matplotlib.pyplot.show()
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}_residuals.png".format(self.get_obfuscated_name(action), degree)), dpi=100)
    matplotlib.pyplot.close(fig)

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
    predict_x = df.ScaledYear.max()
    predicted = func(predict_x)
    r_data["Predicted"][(self.get_obfuscated_name(action), degree)] = predicted
    if self.options.verbose:
      print("Predicted in {} years ({}): {}".format(predict, end, predicted))
    derivative = numpy.poly1d(list(reversed(model.params.values.tolist()))).deriv()
    r_data["PredictedDerivative"][(self.get_obfuscated_name(action), degree)] = derivative(predict_x)
    matplotlib.pyplot.plot([end], [predicted], "cD") # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    ax.add_artist(matplotlib.offsetbox.AnchoredText("$x^{}$; $\\barR^2={:0.3f}$; $y({})\\approx${:0.1f}".format(degree, model.rsquared_adj, end, predicted), loc="upper center"))

    fig = matplotlib.pyplot.gcf()
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha="right", which="both")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    #fig.set_size_inches(10, 5)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}_model.png".format(self.get_obfuscated_name(action), degree)), dpi=100)
    self.write_spreadsheet(df, "{}".format(self.get_obfuscated_name(action)))
    
    if self.options.show_graphs:
      matplotlib.pyplot.show()
      
    matplotlib.pyplot.close(fig)
