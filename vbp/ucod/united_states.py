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
import statsmodels.tsa.api
import matplotlib.offsetbox
import statsmodels.stats.api
import statsmodels.formula.api
import statsmodels.stats.diagnostic
import statsmodels.tools.eval_measures

class UnderlyingCausesOfDeathUnitedStates(vbp.DataSource):
  def initialize_parser(self, parser):
    parser.add_argument("cause", nargs="*", help="ICD Sub-Chapter")
    parser.add_argument("-f", "--file", help="path to file", default="data/ucod/united_states/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
    parser.add_argument("--ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_true")
    parser.add_argument("--no-ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_false")
    parser.add_argument("--ols", help="Ordinary least squares", dest="ols", action="store_true")
    parser.add_argument("--no-ols", help="Ordinary least squares", dest="ols", action="store_false")
    parser.add_argument("--ols-min-degrees", help="minimum polynomial degree", type=int, default=1)
    parser.add_argument("--ols-max-degrees", help="maximum polynomial degree", type=int, default=1)
    parser.set_defaults(
      ets=True,
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
  
  def run_get_action_data(self, action):
    df = self.data[self.data[self.get_action_column_name()] == action]
    df = df.set_index("Date")
    # ETS requires a specific frequency so we forward-fill any missing
    # years. If there's already an anual frequency, no change is made
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    df = df.resample("AS").ffill()
    return df
  
  def set_or_append(self, df, append):
    return append if df is None else pandas.concat([df, append], sort=False)

  def run_predict(self):
    causes = self.options.cause
    if causes is None or len(causes) == 0:
      causes = self.get_possible_actions()
      
    count = len(causes)
    if self.options.verbose:
      print("{} actions".format(count))
    
    model_results = None
    
    action_count = 1
    for cause in causes:
      self.check_action_exists(cause)
      try:
        if self.options.ols:
          for i in range(self.options.ols_min_degrees, self.options.ols_max_degrees + 1):
            model_results = self.set_or_append(model_results, self.create_model_ols(cause, i, self.options.predict, action_count, count))
        if self.options.ets:
          model_results = self.set_or_append(model_results, self.create_model_ets(cause, self.options.predict, action_count, count))
      except:
        if self.options.verbose:
          print(cause)
        traceback.print_exc()
        break
      action_count += 1
      
    if model_results is not None and len(model_results) > 0:
      
      self.write_spreadsheet(model_results, self.prefix_all("model_results"))
      
      m = self.find_best_fitting_models(model_results)
      
      m["S1"] = 1.0
      
      # Adjust any out-of-bounds values
      #r["S2"] = r.apply(lambda row: 0.1 if row.AdjustedR2 < 0.1 else 1.0, axis=1)
      #r["S3"] = r.apply(lambda row: 0.1 if row.ProbFStatistic > 0.05 else 1.0, axis=1)
      #r["S4"] = r.apply(lambda row: 0.1 if row.BreuschPagan < 0.05 else 1.0, axis=1)
      
      if len(m) > 1:
        m["S3"] = vbp.normalize(m.PredictedDerivative.values, 0.5, 1.0)
      else:
        m["S3"] = 1.0
      
      self.write_spreadsheet(m, self.prefix_all("m"))

      b = m[["Predicted", "S1", "S3"]]
      
      self.write_spreadsheet(b, self.prefix_all("b"))
      return b

    else:
      return None
  
  def get_model_map_base(self):
    return {
      "ModelType": {},
      "AIC": {},
      "AICc": {},
      "BIC": {},
      "SSE": {},
      "PredictedYear": {},
      "Predicted": {},
      "PredictedDerivative": {},
    }
  
  def create_model_ets(self, action, predict, i, count):
    print("Creating ETS model {} of {} for: {}".format(i, count, self.get_obfuscated_name(action)))
    df = self.run_get_action_data(action)
    
    df = df[["Crude Rate"]]
    ax = df.plot(color="black", marker="o", legend=True, title="Deaths from {}".format(self.get_obfuscated_name(action)))
    ax.set_ylabel("Crude Rate")
    fig = matplotlib.pyplot.gcf()
    
    model_map = self.get_model_map_base()
    
    results = {}
    results.update(self.run_ets(df["Crude Rate"], color="red", predict=predict, exponential=False, damped=False))
    results.update(self.run_ets(df["Crude Rate"], color="cyan", predict=predict, exponential=False, damped=True))
    results.update(self.run_ets(df["Crude Rate"], color="green", predict=predict, exponential=True, damped=False))
    results.update(self.run_ets(df["Crude Rate"], color="blue", predict=predict, exponential=True, damped=True))
    
    for title, result in results.items():
      index = (self.get_obfuscated_name(action), title)
      model_map["ModelType"][index] = "ETS"
      model_map["PredictedYear"][index] = result[1].index[-1].year
      model_map["Predicted"][index] = result[1][-1]
      model_map["PredictedDerivative"][index] = result[2]
      model_map["AIC"][index] = result[0].aic
      model_map["AICc"][index] = result[0].aicc
      model_map["BIC"][index] = result[0].bic
      model_map["SSE"][index] = result[0].sse
      
    resultdf = pandas.DataFrame(model_map)
    resultdf.index.rename([self.obfuscated_column_name, "Model"], inplace=True)
    
    # https://stackoverflow.com/questions/54791323/
    matplotlib.pyplot.legend()

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(self.create_output_name("{}_ets.png".format(self.get_obfuscated_name(action))), dpi=100)
    
    if self.options.show_graphs:
      matplotlib.pyplot.show()

    matplotlib.pyplot.close(fig)
    
    for title, result in results.items():
      fig, ax = matplotlib.pyplot.subplots()
      ax.scatter(result[0].fittedvalues, result[0].resid)
      ax.set_title("Residuals of {} (${}$)".format(self.get_obfuscated_name(action), title))
      matplotlib.pyplot.tight_layout()
      matplotlib.pyplot.savefig(self.create_output_name("{}_{}_ets_residuals.png".format(self.get_obfuscated_name(action), title)), dpi=100)
      matplotlib.pyplot.close(fig)
      
    self.write_spreadsheet(df, "{}_etsdata".format(self.get_obfuscated_name(action)))
    self.write_spreadsheet(resultdf, "{}_etsresults".format(self.get_obfuscated_name(action)))
    return resultdf
      
  def run_ets(self, df, color, predict, exponential=False, damped=False, damping_slope=0.98):
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.Holt.html
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html
    # https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
    
    if not damped:
      damping_slope = None
      
    fit = statsmodels.tsa.api.Holt(df, exponential=exponential, damped=damped).fit(damping_slope=damping_slope)
    fit.fittedvalues.plot(color=color, style="--", label="_nolegend_")
    title = "ETS(A,"
    title += "M" if exponential else "A"
    title += "_d" if damped else ""
    title += ",N)"
    forecast = fit.forecast(predict).rename("${}$".format(title))
    forecast.plot(color=color, legend=True, style="--")
    
    # Not necessarily linear, so take the slope of a line through the last two points
    lasttwopoints = forecast.iloc[-2::,]
    slope = (lasttwopoints.iloc[1,] - lasttwopoints.iloc[0,]) / (lasttwopoints.index[1].year - lasttwopoints.index[0].year)
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(list(map(lambda x: x.year, forecast.index.to_pydatetime())), forecast.values)
    
    return {title: [fit, forecast, slope]}

  def create_model_ols(self, action, degree, predict, i, count):
    df = self.run_get_action_data(action)
    df = df.reset_index()
    
    model_map = self.get_model_map_base()
    model_map.update({
      "AdjustedR2": {},
      "ProbFStatistic": {},
      "LogLikelihood": {},
      "BreuschPagan": {},
      "Formula": {},
    })

    print("Creating OLS model {} of {} for: {}, degrees: {}".format(i, count, self.get_obfuscated_name(action), degree))
      
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
    
    index = (self.get_obfuscated_name(action), degree)
    model_map["ModelType"][index] = "OLS"
    model_map["AdjustedR2"][index] = model.rsquared_adj
    model_map["ProbFStatistic"][index] = model.f_pvalue
    model_map["LogLikelihood"][index] = model.llf
    model_map["AIC"][index] = model.aic
    model_map["BIC"][index] = model.bic
    model_map["Formula"][index] = vbp.linear_regression_modeled_formula(model, degree)
    
    model_map["AICc"][index] = statsmodels.tools.eval_measures.aicc(model.llf, model.nobs, degree + 1)
    
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
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}_ols_residuals.png".format(self.get_obfuscated_name(action), degree)), dpi=100)
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
    model_map["BreuschPagan"][index] = f_pvalue

    ax = df.plot("Year", "Crude Rate", grid=True, title="Deaths from {}".format(self.get_obfuscated_name(action)), color="black", marker="o")

    func = numpy.polynomial.Polynomial(model.params)
    matplotlib.pyplot.plot(df["Year"], func(df["ScaledYear"]), "--", color="blue", label="OLS $x^{}$".format(degree))
    
    end -= 1
    predict_x = df.ScaledYear.max()
    predicted = func(predict_x)
    model_map["Predicted"][index] = predicted
    model_map["PredictedYear"][index] = end
    if self.options.verbose:
      print("Predicted in {} years ({}): {}".format(predict, end, predicted))
    derivative = numpy.poly1d(list(reversed(model.params.values.tolist()))).deriv()
    model_map["PredictedDerivative"][index] = derivative(predict_x)
    matplotlib.pyplot.plot([end], [predicted], "cD") # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    ax.add_artist(matplotlib.offsetbox.AnchoredText("$x^{}$; $\\barR^2={:0.3f}$; $y({})\\approx${:0.1f}".format(degree, model.rsquared_adj, end, predicted), loc="upper center"))

    fig = matplotlib.pyplot.gcf()
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha="right", which="both")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    #fig.set_size_inches(10, 5)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}_ols.png".format(self.get_obfuscated_name(action), degree)), dpi=100)
    
    if self.options.show_graphs:
      matplotlib.pyplot.show()
      
    matplotlib.pyplot.close(fig)

    resultdf = pandas.DataFrame(model_map)
    resultdf.index.rename([self.obfuscated_column_name, "Model"], inplace=True)

    self.write_spreadsheet(df, "{}_olsdata".format(self.get_obfuscated_name(action)))
    self.write_spreadsheet(resultdf, "{}_olsresults".format(self.get_obfuscated_name(action)))
    return resultdf
