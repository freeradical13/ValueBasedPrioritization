import os
import sys
import vbp
import glob
import math
import numpy
import scipy
import pandas
import pprint
import zipfile
import argparse
import datetime
import traceback
import matplotlib
import urllib.request
import matplotlib.pyplot
import statsmodels.tools
import statsmodels.tsa.api
import matplotlib.offsetbox
import statsmodels.stats.api
import statsmodels.formula.api
import statsmodels.stats.diagnostic
import statsmodels.tools.eval_measures

from vbp.ucod.icd import ICD

class UnderlyingCausesOfDeathUnitedStates(vbp.DataSource):

  # Population estimates (some mid-year) used to calculate death rates shown in Vital Statistics of the United States
  # https://www.cdc.gov/nchs/nvss/mortality/historical_population.htm
  # 1900-1959: https://www.cdc.gov/nchs/data/dvs/pop0059.pdf
  # 1960-1997: https://www.cdc.gov/nchs/data/dvs/pop6097.pdf
  # 1998     : https://www2.census.gov/programs-surveys/popest/tables/1900-1980/national/totals/popclockest.txt
  # 1999-2017: https://wonder.cdc.gov/ucd-icd10.html
  mortality_uspopulation_years = list(range(1900, 2018))
  mortality_uspopulation_per_year = [
     19965446,  20237453,  20582907,  20943223,  21332076,  21767980,  33782288,  34552837,  38634758,  44223513, # 1900-1909
     47470437,  53929644,  54847700,  58156738,  60963307,  61894847,  66971177,  70234775,  79008411,  83157982, # 1910-1919
     86079263,  87814447,  92702899,  96788196,  99318098, 102031554, 103822682, 107084531, 113636159, 115317449, # 1920-1929
    117238278, 118148987, 118903899, 125578763, 126373773, 127250232, 128053180, 128824829, 129824939, 130879718, # 1930-1939
    131669275, 133121000, 133920000, 134245000, 132885000, 132481000, 140054000, 143446000, 146093000, 148665000, # 1940-1949
    150697361, 153310000, 155687000, 158242000, 161164000, 164308000, 167306000, 170371000, 173320000, 176513000, # 1950-1959
    179323175, 182992000, 185771000, 188483000, 191141000, 193526000, 195576000, 197457000, 199399000, 201385000, # 1960-1969
    203211926, 206827000, 209284000, 211357000, 213342000, 215465000, 217563000, 219760000, 222095000, 224567000, # 1970-1979
    226545805, 229466000, 231664000, 233792000, 235825000, 237924000, 240133000, 242289000, 244499000, 246819000, # 1980-1989
    248709873, 252177000, 255077536, 257783004, 260340990, 262755270, 265283783, 267636061, 270248003, 279040168, # 1990-1999
    281421906, 284968955, 287625193, 290107933, 292805298, 295516599, 298379912, 301231207, 304093966, 306771529, # 2000-2009
    308745538, 311591917, 313914040, 316128839, 318857056, 321418820, 323127513, 325719178,                       # 2010-2017
  ]
  mortality_icd_revision = [
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1900-1909
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1910-1919
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1920-1929
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1930-1939
     5,  5,  5,  5,  5,  5,  5,  5,  5,  6, # 1940-1949
     6,  6,  6,  6,  6,  6,  6,  6,  7,  7, # 1950-1959
     7,  7,  7,  7,  7,  7,  7,  7,  8,  8, # 1960-1969
     8,  8,  8,  8,  8,  8,  8,  9,  9,  9, # 1970-1979
     9,  9,  9,  9,  9,  9,  9,  9,  9,  9, # 1980-1989
     9,  9,  9,  9,  9,  9,  9,  9,  9, 10, # 1990-1999
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, # 2000-2009
    10, 10, 10, 10, 10, 10, 10, 10,         # 2010-2017
  ]
  mortality_uspopulation = pandas.DataFrame(
    {"Population": mortality_uspopulation_per_year, "ICDRevision": mortality_icd_revision},
    index=mortality_uspopulation_years
  )
  
  def initialize_parser(self, parser):
    parser.add_argument("cause", nargs="*", help="ICD Sub-Chapter")
    parser.add_argument("-f", "--file", help="path to file", default="data/ucod/united_states/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
    parser.add_argument("-r", "--raw-files-directory", help="directory with raw files")
    parser.add_argument("--download", help="If not files in --raw-files-directory, download and extract", action="store_true", default=True)
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
    self.write_spreadsheet(df, self.prefix_all("data"))
    
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
  
  def get_causes(self):
    causes = self.options.cause
    if causes is None or len(causes) == 0:
      causes = self.get_possible_actions()
    return causes

  def run_predict(self):
    causes = self.get_causes()
      
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

      b = m[[self.predict_column_name, "S1", "S3"]]
      
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
      self.predict_column_name + "Year": {},
      self.predict_column_name: {},
      self.predict_column_name + "Derivative": {},
    }
  
  def create_model_ets(self, action, predict, i, count):
    print("Creating ETS model {} of {} for: {}".format(i, count, self.get_obfuscated_name(action)))
    df = self.run_get_action_data(action)
    
    df = df[["Crude Rate"]]
    ax = df.plot(color="black", marker="o", legend=True, title="Deaths from {}".format(self.get_obfuscated_name(action)), kind="line")
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
      model_map[self.predict_column_name + "Year"][index] = result[1].index[-1].year
      model_map[self.predict_column_name][index] = result[1][-1]
      model_map[self.predict_column_name + "Derivative"][index] = result[2]
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

    ax = df.plot("Year", "Crude Rate", grid=True, title="Deaths from {}".format(self.get_obfuscated_name(action)), color="black", marker="o", kind="line", legend=True)

    func = numpy.polynomial.Polynomial(model.params)
    matplotlib.pyplot.plot(df["Year"], func(df["ScaledYear"]), "--", color="blue", label="OLS $x^{}$".format(degree))
    
    end -= 1
    predict_x = df.ScaledYear.max()
    predicted = func(predict_x)
    model_map[self.predict_column_name][index] = predicted
    model_map[self.predict_column_name + "Year"][index] = end
    if self.options.verbose:
      print("{} in {} years ({}): {}".format(self.predict_column_name, predict, end, predicted))
    derivative = numpy.poly1d(list(reversed(model.params.values.tolist()))).deriv()
    model_map[self.predict_column_name + "Derivative"][index] = derivative(predict_x)
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

  def run_prophet(self):
    import fbprophet
    causes = self.get_causes()
    for cause in causes:
      df = self.get_action_data(cause)
      df.reset_index(inplace=True)
      df = df[["Date", "Crude Rate"]]
      df.rename(columns={"Date": "ds", "Crude Rate": "y"}, inplace=True)
      # https://facebook.github.io/prophet/docs/saturating_forecasts.html
      df["floor"] = 0
      df["cap"] = 100000
      prophet = fbprophet.Prophet()
      prophet.fit(df)
      future = prophet.make_future_dataframe(periods=10, freq="Y")
      forecast = prophet.predict(future)

      prophet.plot(forecast)
      
      prophet.plot_components(forecast)

      matplotlib.pyplot.show()
      
      forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
      print(forecast)
      
  def download_raw_files(self):
    print("Downloading raw files from https://www.nber.org/data/vital-statistics-mortality-data-multiple-cause-of-death.html")
    if not os.path.exists(self.options.raw_files_directory):
      os.makedirs(self.options.raw_files_directory)
    
    for i in range(1959, 2018):
      print("Downloading {}...".format(i))
      downloaded_file = os.path.join(self.options.raw_files_directory, "mort{0}.csv.zip".format(i))
      urllib.request.urlretrieve("https://www.nber.org/mortality/{0}/mort{0}.csv.zip".format(i), downloaded_file)
      with zipfile.ZipFile(downloaded_file, "r") as zfile:
        print("Unzipping mort{0}.csv.zip".format(i))
        zfile.extractall(self.options.raw_files_directory)
        os.remove(downloaded_file)
      
    raise ValueError("")

  def run_generate_average_ages(self):
    if self.options.raw_files_directory is None:
      raise ValueError("--raw-files-directory required")
    if not os.path.exists(self.options.raw_files_directory):
      if self.options.download:
        self.download_raw_files()
      else:
        raise ValueError("--raw-files-directory does not exist")
    if not os.path.isdir(self.options.raw_files_directory):
      raise ValueError("--raw-files-directory is not a directory")
    
    csvs = glob.glob(os.path.join(self.options.raw_files_directory, "*.csv"))
    if len(csvs) == 0:
      if self.options.download:
        self.download_raw_files()
      else:
        raise ValueError("--raw-files-directory does not have any .csv files. See https://www.nber.org/data/vital-statistics-mortality-data-multiple-cause-of-death.html")
    
    # usecols=["age", "year", "ucod", "ucr358", "ucr113", "ucr130", "ucr39"]
    df = pandas.read_csv(csvs[0], dtype={"age": str})
    df["AgeMinutes"] = df["age"].apply(ICD.convert_age_minutes)
    #print(df.AgeMinutes.max())
    print(df)
    
    #causes = self.get_causes()
    #for cause in causes:
      #print(cause)

  def run_test(self):
    counts = {}
    csvs = glob.glob(os.path.join(self.options.raw_files_directory, "*.csv"))
    for csv in csvs:
      filename, file_extension = os.path.splitext(os.path.basename(csv))
      if filename.startswith("mort"):
        filename = filename[4:]
        file_year = int(filename)
        count_years = {
          "Total Deaths": 0,
          "ICD Revision": 0,
          "Influenza and pneumonia": numpy.NaN,
          "Tuberculosis": numpy.NaN,
          "Diarrhea, enteritis, and colitis": numpy.NaN,
          "Heart disease": numpy.NaN,
          "Stroke": numpy.NaN,
          "Kidney disease": numpy.NaN,
          "Accidents excluding motor vehicles": numpy.NaN,
          "Cancer": numpy.NaN,
          "Perinatal Conditions": numpy.NaN,
          "Diabetes": numpy.NaN,
          "Motor vehicle accidents": numpy.NaN,
          "Arteriosclerosis": numpy.NaN,
          "Congenital Malformations": numpy.NaN,
          "Cirrhosis of liver": numpy.NaN,
        }
        counts[file_year] = count_years

        count_years["ICD Revision"] = UnderlyingCausesOfDeathUnitedStates.mortality_uspopulation.loc[file_year]["ICDRevision"]
        print("Processing {}".format(file_year))
        
        if True:
          yearcol = "datayear" if file_year <= 1995 else "year"
          df = pandas.read_csv(
            csv,
            usecols=[yearcol, "age", "ucod"],
            dtype={
              yearcol: numpy.int32,
              "age": str,
              "ucod": str,
            },
            na_values=["&", "-"]
          )
          if file_year >= 1968 and file_year <= 1977:
            df[yearcol] = df[yearcol].apply(lambda x: x + 1960 if x >= 8 else x + 1970)
          if file_year == 1978:
            df[yearcol] = df[yearcol].apply(lambda x: x + 1970)
          if file_year >= 1979 and file_year <= 1995:
            df[yearcol] = df[yearcol].apply(lambda x: x + 1900)
          if df[yearcol].min() != file_year or df[yearcol].max() != file_year:
            raise ValueError("Unexpected year value {} in data for {}".format(df[yearcol].min(), csv))
          
          df["AgeMinutes"] = df["age"].apply(ICD.convert_age_minutes)
          df["ucodint"] = df["ucod"].apply(ICD.toint)
          df["ucodfloat"] = df["ucod"].apply(ICD.tofloat)
          
          scale = 1
          if file_year == 1972:
            # "The 1972 file is a 50% sample" (https://www.nber.org/mortality/errata.txt)
            # Same in raw data: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/mort1972us.zip
            scale = 2
          
          count_years["Total Deaths"] = len(df)*scale

          if file_year >= 1959 and file_year <= 1967:
            count_years["Tuberculosis"] = len(df[df["ucodint"].isin(list(range(1, 20)))])*scale
            count_years["Diarrhea, enteritis, and colitis"] = len(df[df["ucodint"].isin([543, 571, 572])])*scale
            count_years["Cancer"] = len(df[df["ucodint"].isin(list(range(140, 206)))])*scale
            count_years["Diabetes"] = len(df[df["ucodint"] == 260])*scale
            count_years["Heart disease"] = len(df[df["ucodint"].isin([402, 404] + list(range(410, 444)))])*scale
            count_years["Stroke"] = len(df[df["ucodint"].isin(list(range(330, 335)))])*scale
            count_years["Arteriosclerosis"] = len(df[df["ucodint"] == 450])*scale
            count_years["Influenza and pneumonia"] = len(df[df["ucodint"].isin(list(range(480, 494)))])*scale
            count_years["Cirrhosis of liver"] = len(df[df["ucodint"] == 581])*scale
            count_years["Kidney disease"] = len(df[df["ucodint"].isin(list(range(590, 595)))])*scale
            count_years["Congenital Malformations"] = len(df[df["ucodint"].isin(list(range(750, 760)))])*scale
            count_years["Perinatal Conditions"] = len(df[df["ucodint"].isin(list(range(760, 777)))])*scale
            count_years["Motor vehicle accidents"] = len(df[df["ucodint"].isin(list(range(810, 836)))])*scale
            count_years["Accidents excluding motor vehicles"] = len(df[df["ucodint"].isin(list(range(800, 803)) + list(range(840, 963)))])*scale
          elif file_year >= 1968 and file_year <= 1976:
            count_years["Tuberculosis"] = len(df[df["ucodint"].isin(list(range(10, 20)))])*scale
            count_years["Diarrhea, enteritis, and colitis"] = len(df[df["ucodint"] == 9])*scale
            count_years["Cancer"] = len(df[df["ucodint"].isin(list(range(140, 210)))])*scale
            count_years["Diabetes"] = len(df[df["ucodint"] == 250])*scale
            count_years["Heart disease"] = len(df[df["ucodint"].isin(list(range(390, 399)) + [402, 404] + list(range(410, 430)))])*scale
            count_years["Stroke"] = len(df[df["ucodint"].isin(list(range(430, 439)))])*scale
            count_years["Arteriosclerosis"] = len(df[df["ucodint"] == 440])*scale
            count_years["Influenza and pneumonia"] = len(df[df["ucodint"].isin(list(range(470, 475))+list(range(480, 487)))])*scale
            count_years["Cirrhosis of liver"] = len(df[df["ucodint"] == 571])*scale
            count_years["Kidney disease"] = len(df[df["ucodint"].isin(list(range(580, 585)))])*scale
            count_years["Congenital Malformations"] = len(df[df["ucodint"].isin(list(range(740, 750)))])*scale
            count_years["Perinatal Conditions"] = len(df[((df["ucodfloat"] >= 760) & (df["ucodfloat"] <= 769.2)) | ((df["ucodfloat"] >= 769.4) & (df["ucodfloat"] <= 772)) | ((df["ucodfloat"] >= 774) & (df["ucodfloat"] <= 778))])*scale
            count_years["Motor vehicle accidents"] = len(df[df["ucodint"].isin(list(range(810, 824)))])*scale
            count_years["Accidents excluding motor vehicles"] = len(df[df["ucodint"].isin(list(range(800, 808)) + list(range(825, 950)))])*scale
          elif file_year >= 1977 and file_year <= 1998:
            count_years["Tuberculosis"] = len(df[df["ucodint"].isin(list(range(10, 19)))])*scale
            count_years["Diarrhea, enteritis, and colitis"] = len(df[df["ucodint"] == 9])*scale
            count_years["Cancer"] = len(df[df["ucodint"].isin(list(range(140, 209)))])*scale
            count_years["Diabetes"] = len(df[df["ucodint"] == 250])*scale
            count_years["Heart disease"] = len(df[df["ucodint"].isin(list(range(390, 399)) + [402, 404] + list(range(410, 430)))])*scale
            count_years["Stroke"] = len(df[df["ucodint"].isin(list(range(430, 439)))])*scale
            count_years["Arteriosclerosis"] = len(df[df["ucodint"] == 440])*scale
            count_years["Influenza and pneumonia"] = len(df[df["ucodint"].isin(list(range(480, 488)))])*scale
            count_years["Cirrhosis of liver"] = len(df[df["ucodint"] == 571])*scale
            count_years["Kidney disease"] = len(df[df["ucodint"].isin(list(range(580, 590)))])*scale
            count_years["Congenital Malformations"] = len(df[df["ucodint"].isin(list(range(740, 750)))])*scale
            count_years["Perinatal Conditions"] = len(df[df["ucodint"].isin(list(range(760, 780)))])*scale
            count_years["Motor vehicle accidents"] = len(df[df["ucodint"].isin(list(range(810, 826)))])*scale
            count_years["Accidents excluding motor vehicles"] = len(df[df["ucodint"].isin(list(range(800, 808)) + list(range(826, 950)))])*scale
          elif file_year >= 1999:
            df.to_csv("test.csv")
            count_years["Tuberculosis"] = len(df[df["ucodint"].isin(list(range(ICD.toint("A16"), ICD.toint("A20"))))])*scale
            count_years["Diarrhea, enteritis, and colitis"] = len(df[df["ucodint"] == ICD.toint("A09")])*scale
            count_years["Cancer"] = len(df[df["ucodint"].isin(list(range(ICD.toint("C00"), ICD.toint("C98"))))])*scale
            count_years["Diabetes"] = len(df[df["ucodint"].isin(list(range(ICD.toint("E10"), ICD.toint("E15"))))])*scale
            count_years["Heart disease"] = len(df[df["ucodint"].isin(list(range(ICD.toint("I00"), ICD.toint("I10"))) + [ICD.toint("I11"), ICD.toint("I13")] + list(range(ICD.toint("I20"), ICD.toint("I52"))))])*scale
            count_years["Stroke"] = len(df[df["ucodint"].isin(list(range(ICD.toint("I60"), ICD.toint("I70"))) + [ICD.toint("G45")])])*scale
            count_years["Arteriosclerosis"] = len(df[df["ucodint"] == ICD.toint("I70")])*scale
            count_years["Influenza and pneumonia"] = len(df[df["ucodint"].isin(list(range(ICD.toint("J10"), ICD.toint("J19"))))])*scale
            count_years["Cirrhosis of liver"] = len(df[df["ucodint"].isin(list(range(ICD.toint("K73"), ICD.toint("K75"))) + [ICD.toint("K70")])])*scale
            count_years["Kidney disease"] = len(df[df["ucodint"].isin(list(range(ICD.toint("N00"), ICD.toint("N08"))) + list(range(ICD.toint("N17"), ICD.toint("N20"))) + list(range(ICD.toint("N25"), ICD.toint("N28"))))])*scale
            count_years["Congenital Malformations"] = len(df[df["ucodint"].isin(list(range(ICD.toint("Q00"), ICD.toint("Q99", addone=True))))])*scale
            count_years["Perinatal Conditions"] = len(df[df["ucodint"].isin(list(range(ICD.toint("P00"), ICD.toint("P97"))) + [ICD.toint("A33")])])*scale
            count_years["Motor vehicle accidents"] = len(df[((df["ucodfloat"] >= ICD.tofloat("V02")) & (df["ucodfloat"] <= ICD.tofloat("V04"))) | ((df["ucodfloat"] >= ICD.tofloat("V12")) & (df["ucodfloat"] <= ICD.tofloat("V14"))) | ((df["ucodfloat"] >= ICD.tofloat("V19.0")) & (df["ucodfloat"] <= ICD.tofloat("V19.2"))) | ((df["ucodfloat"] >= ICD.tofloat("V19.4")) & (df["ucodfloat"] <= ICD.tofloat("V19.6"))) | ((df["ucodfloat"] >= ICD.tofloat("V20")) & (df["ucodfloat"] <= ICD.tofloat("V79"))) | ((df["ucodfloat"] >= ICD.tofloat("V80.3")) & (df["ucodfloat"] <= ICD.tofloat("V80.5"))) | ((df["ucodfloat"] >= ICD.tofloat("V81.0")) & (df["ucodfloat"] <= ICD.tofloat("V81.1"))) | ((df["ucodfloat"] >= ICD.tofloat("V82.0")) & (df["ucodfloat"] <= ICD.tofloat("V82.1"))) | ((df["ucodfloat"] >= ICD.tofloat("V83")) & (df["ucodfloat"] <= ICD.tofloat("V86"))) | ((df["ucodfloat"] >= ICD.tofloat("V87.0")) & (df["ucodfloat"] <= ICD.tofloat("V87.8"))) | ((df["ucodfloat"] >= ICD.tofloat("V88.0")) & (df["ucodfloat"] <= ICD.tofloat("V88.8"))) | (df["ucodint"] == ICD.toint("V09")) | (df["ucodint"] == ICD.toint("V89")) | (df["ucodfloat"] == ICD.tofloat("V09.2")) | (df["ucodfloat"] == ICD.tofloat("V89.2"))])*scale
            count_years["Accidents excluding motor vehicles"] = len(df[((df["ucodfloat"] >= ICD.tofloat("V05")) & (df["ucodfloat"] <= ICD.tofloat("V08"))) | ((df["ucodfloat"] >= ICD.tofloat("V09.3")) & (df["ucodfloat"] <= ICD.tofloat("V11"))) | ((df["ucodfloat"] >= ICD.tofloat("V15")) & (df["ucodfloat"] <= ICD.tofloat("V18"))) | ((df["ucodfloat"] >= ICD.tofloat("V19.7")) & (df["ucodfloat"] <= ICD.tofloat("V19.9"))) | ((df["ucodfloat"] >= ICD.tofloat("V80.0")) & (df["ucodfloat"] <= ICD.tofloat("V80.2"))) | ((df["ucodfloat"] >= ICD.tofloat("V80.6")) & (df["ucodfloat"] <= ICD.tofloat("V80.9"))) | ((df["ucodfloat"] >= ICD.tofloat("V81.2")) & (df["ucodfloat"] <= ICD.tofloat("V81.9"))) | ((df["ucodfloat"] >= ICD.tofloat("V82.2")) & (df["ucodfloat"] <= ICD.tofloat("V82.9"))) | ((df["ucodfloat"] >= ICD.tofloat("V89.3")) & (df["ucodfloat"] <= ICD.tofloat("X59"))) | ((df["ucodfloat"] >= ICD.tofloat("Y85")) & (df["ucodfloat"] <= ICD.tofloat("Y86"))) | (df["ucodfloat"] == ICD.tofloat("V01")) | (df["ucodfloat"] == ICD.tofloat("V09.1")) | (df["ucodfloat"] == ICD.tofloat("V19.3")) | (df["ucodfloat"] == ICD.tofloat("V87.9")) | (df["ucodfloat"] == ICD.tofloat("V88.9")) | (df["ucodfloat"] == ICD.tofloat("V89.1"))])*scale

    df = pandas.DataFrame.from_dict(counts, orient="index")
    df.to_csv("tmp.csv")
    print(df[["Total Deaths", "ICD Revision", "Heart disease", "Cancer", "Stroke", "Kidney disease"]])
    
