import os
import sys
import vbp
import enum
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

class DataType(enum.Enum):
  UCOD_1999_2017_SUB_CHAPTERS = enum.auto()
  UCOD_LONGTERM_COMPARABLE_LEADING = enum.auto()
  
  def __str__(self):
    return self.name
  
  @classmethod
  def _missing_(cls, name):
    for member in cls:
      if member.name.lower() == name.lower():
        return member

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
  
  # https://www.cdc.gov/nchs/data/nvsr/nvsr49/nvsr49_02.pdf
  mortality_icd_revision = [
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1900-1909
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1910-1919
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1920-1929
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5, # 1930-1939
     5,  5,  5,  5,  5,  5,  5,  5,  5,  6, # 1940-1949; Switch @ 1949
     6,  6,  6,  6,  6,  6,  6,  6,  7,  7, # 1950-1959; Switch @ 1958
     7,  7,  7,  7,  7,  7,  7,  7,  8,  8, # 1960-1969; Switch @ 1968
     8,  8,  8,  8,  8,  8,  8,  8,  8,  9, # 1970-1979; Switch @ 1979
     9,  9,  9,  9,  9,  9,  9,  9,  9,  9, # 1980-1989
     9,  9,  9,  9,  9,  9,  9,  9,  9, 10, # 1990-1999; Switch @ 1999
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, # 2000-2009
    10, 10, 10, 10, 10, 10, 10, 10,         # 2010-2017
  ]
  
  mortality_uspopulation = pandas.DataFrame(
    {"Population": mortality_uspopulation_per_year, "ICDRevision": mortality_icd_revision},
    index=mortality_uspopulation_years
  )
  
  def initialize_parser(self, parser):
    parser.add_argument("cause", nargs="*", help="ICD Sub-Chapter")
    parser.add_argument("--average-ages", help="Compute average ages column with the specified column name", default="AverageAge")
    parser.add_argument("--average-age-range", help="Range over which to calculate the average age", type=int, default=5)
    parser.add_argument("--comparable-ratios", help="Process comparable ratios for raw mortality matrix for prepare_data", action="store_true", default=False)
    parser.add_argument("--comparable-ratios-input-file", help="Comparable ratios file", default="data/ucod/united_states/comparable_ucod_estimates.xlsx")
    parser.add_argument("--comparable-ratios-output-file", help="Comparable ratios file", default="data/ucod/united_states/comparable_ucod_estimates_ratios_applied.xlsx")
    parser.add_argument("--download", help="If not files in --raw-files-directory, download and extract", action="store_true", default=True)
    parser.add_argument("--ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_true")
    parser.add_argument("-f", "--file", help="path to file", default="data/ucod/united_states/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
    parser.add_argument("--no-ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_false")
    parser.add_argument("--ols", help="Ordinary least squares", dest="ols", action="store_true")
    parser.add_argument("--no-ols", help="Ordinary least squares", dest="ols", action="store_false")
    parser.add_argument("--ols-min-degrees", help="minimum polynomial degree", type=int, default=1)
    parser.add_argument("--ols-max-degrees", help="maximum polynomial degree", type=int, default=1)
    parser.add_argument("--raw-files-directory", help="directory with raw files", default="data/ucod/united_states/mort/")
    parser.add_argument("--test", help="Test")
    parser.set_defaults(
      ets=True,
    )

  @staticmethod
  def get_data_types_enum():
    return DataType

  @staticmethod
  def get_data_types_enum_default():
    return DataType.UCOD_1999_2017_SUB_CHAPTERS

  def run_load(self):
    if self.options.data_type == DataType.UCOD_1999_2017_SUB_CHAPTERS:
      df = pandas.read_csv(
            self.options.file,
            sep="\t",
            usecols=["Year", "ICD Sub-Chapter", "ICD Sub-Chapter Code", "Deaths", "Population"],
            na_values=["Unreliable"],
            parse_dates=[0]
          ).dropna(how="all")
    elif self.options.data_type == DataType.UCOD_LONGTERM_COMPARABLE_LEADING:
      df = pandas.read_excel(
        self.options.comparable_ratios_output_file,
        index_col=0,
        parse_dates=[0],
      )
      df.drop(columns=["Total Deaths", "ICD Revision"], inplace=True)
      melt_cols = df.columns.values
      df = df.reset_index().melt(id_vars=["Year"], value_vars=melt_cols, var_name=self.get_action_column_name(), value_name="Deaths").sort_values(by=["Year", self.get_action_column_name()])
      df.Year = df.Year
      df["Population"] = df.Year.apply(lambda y: self.mortality_uspopulation.loc[y.year]["Population"])
    else:
      raise NotImplementedError()
    
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
      
      if len(m) > 1:
        m["S3"] = vbp.normalize(m.PredictedDerivative.values, 0.5, 1.0)
      else:
        m["S3"] = 1.0
      
      self.write_spreadsheet(m, self.prefix_all("m"))

      extra_columns = ["S1", "S3"]
      final_columns = [self.predict_column_name]

      extra_columns.sort()
      final_columns += extra_columns
      b = m[final_columns]
      
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
    ax = df.plot(color="black", marker="o", legend=True, title="Deaths from {}".format(self.get_obfuscated_name(action)), grid=True, kind="line")
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
    forecast.plot(color=color, legend=True, grid=True, style="--")
    
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
  
  def check_raw_files_directory(self):
    if self.options.raw_files_directory is None:
      raise ValueError("--raw-files-directory required")
    if not os.path.exists(self.options.raw_files_directory):
      if self.options.download:
        self.download_raw_files()
      else:
        raise ValueError("--raw-files-directory does not exist")
    if not os.path.isdir(self.options.raw_files_directory):
      raise ValueError("--raw-files-directory is not a directory")

  def run_prepare_data(self):
    self.check_raw_files_directory()
    if self.options.comparable_ratios:
      self.create_comparable()
    else:
      self.process_raw_mortality_data()
    
  def get_mortality_files(self):
    return glob.glob(os.path.join(self.options.raw_files_directory, "*.csv"))
  
  def get_mortality_file_info(self, csv):
    filename, file_extension = os.path.splitext(os.path.basename(csv))
    if filename.startswith("mort"):
      filename = filename[4:]
      file_year = int(filename)
      return filename, file_extension, file_year
    else:
      return None, None, None
    
  def get_mortality_data(self, csv, file_year):
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
    
    return df, scale
    
  def process_raw_mortality_data(self):
    counts = {}
    csvs = self.get_mortality_files()
    for i, csv in enumerate(csvs):
      filename, file_extension, file_year = self.get_mortality_file_info(csv)
      if filename:
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
          "Typhoid fever": numpy.NaN,
          "Measles": numpy.NaN,
          "Whooping cough": numpy.NaN,
          "Diphtheria": numpy.NaN,
          "Intestinal infections": numpy.NaN,
          "Meningococcal infections": numpy.NaN,
          "Acute poliomyelitis": numpy.NaN,
          "Syphilis": numpy.NaN,
          "Acute rheumatic fever": numpy.NaN,
          "Hypertension": numpy.NaN,
          "Chronic respiratory diseases": numpy.NaN,
          "Ulcer": numpy.NaN,
          "Suicide": numpy.NaN,
          "Homicide": numpy.NaN,
        }
        counts[file_year] = count_years

        count_years["ICD Revision"] = UnderlyingCausesOfDeathUnitedStates.mortality_uspopulation.loc[file_year]["ICDRevision"]
        self.print_processing_csv(i, csv, csvs)
        
        df, scale = self.get_mortality_data(csv, file_year)
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
          count_years["Typhoid fever"] = len(df[df["ucodint"] == 40])*scale
          count_years["Measles"] = len(df[df["ucodint"] == 85])*scale
          count_years["Whooping cough"] = len(df[df["ucodint"] == 56])*scale
          count_years["Diphtheria"] = len(df[df["ucodint"] == 55])*scale
          count_years["Intestinal infections"] = len(df[df["ucodint"].isin([571, 764])])*scale
          count_years["Meningococcal infections"] = len(df[df["ucodint"] == 57])*scale
          count_years["Acute poliomyelitis"] = len(df[df["ucodint"] == 80])*scale
          count_years["Syphilis"] = len(df[df["ucodint"].isin(list(range(20, 30)))])*scale
          count_years["Acute rheumatic fever"] = len(df[df["ucodint"].isin(list(range(400, 403)))])*scale
          count_years["Hypertension"] = len(df[df["ucodint"].isin(list(range(444, 448)))])*scale
          count_years["Chronic respiratory diseases"] = len(df[(df["ucodint"] == 241) | (df["ucodint"] == 501) | (df["ucodint"] == 502) | (numpy.isclose(df["ucodfloat"], 527.1))])*scale
          count_years["Ulcer"] = len(df[df["ucodint"].isin(list(range(540, 542)))])*scale
          count_years["Suicide"] = len(df[df["ucodint"].isin([963] + list(range(970, 980)))])*scale
          count_years["Homicide"] = len(df[df["ucodint"].isin([964] + list(range(980, 986)))])*scale
        elif file_year >= 1968 and file_year <= 1978:
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
          count_years["Perinatal Conditions"] = len(df[((df["ucodint"] >= 760) & (df["ucodfloat"] < 769.3)) | ((df["ucodfloat"] > 769.3) & (df["ucodint"] <= 772)) | ((df["ucodint"] >= 774) & (df["ucodint"] <= 778))])*scale
          count_years["Motor vehicle accidents"] = len(df[df["ucodint"].isin(list(range(810, 824)))])*scale
          count_years["Accidents excluding motor vehicles"] = len(df[df["ucodint"].isin(list(range(800, 808)) + list(range(825, 950)))])*scale
          count_years["Typhoid fever"] = len(df[df["ucodint"] == 1])*scale
          count_years["Measles"] = len(df[df["ucodint"] == 55])*scale
          count_years["Whooping cough"] = len(df[df["ucodint"] == 33])*scale
          count_years["Diphtheria"] = len(df[df["ucodint"] == 32])*scale
          count_years["Intestinal infections"] = len(df[df["ucodint"].isin([8, 9])])*scale
          count_years["Meningococcal infections"] = len(df[df["ucodint"] == 36])*scale
          count_years["Acute poliomyelitis"] = len(df[df["ucodint"].isin(list(range(40, 44)))])*scale
          count_years["Syphilis"] = len(df[df["ucodint"].isin(list(range(90, 98)))])*scale
          count_years["Acute rheumatic fever"] = len(df[df["ucodint"].isin(list(range(390, 393)))])*scale
          count_years["Hypertension"] = len(df[df["ucodint"].isin([400, 401, 403])])*scale
          count_years["Chronic respiratory diseases"] = len(df[df["ucodint"].isin(list(range(490, 494)))])*scale
          count_years["Ulcer"] = len(df[df["ucodint"].isin(list(range(531, 534)))])*scale
          count_years["Suicide"] = len(df[df["ucodint"].isin(list(range(950, 960)))])*scale
          count_years["Homicide"] = len(df[df["ucodint"].isin(list(range(960, 979)))])*scale
        elif file_year >= 1979 and file_year <= 1998:
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
          count_years["Typhoid fever"] = len(df[numpy.isclose(df["ucodfloat"], 2)])*scale
          count_years["Measles"] = len(df[df["ucodint"] == 55])*scale
          count_years["Whooping cough"] = len(df[df["ucodint"] == 33])*scale
          count_years["Diphtheria"] = len(df[df["ucodint"] == 32])*scale
          count_years["Intestinal infections"] = len(df[df["ucodint"].isin(list(range(7, 10)))])*scale
          count_years["Meningococcal infections"] = len(df[df["ucodint"] == 36])*scale
          count_years["Acute poliomyelitis"] = len(df[df["ucodint"] == 45])*scale
          count_years["Syphilis"] = len(df[df["ucodint"].isin(list(range(90, 98)))])*scale
          count_years["Acute rheumatic fever"] = len(df[df["ucodint"].isin(list(range(390, 393)))])*scale
          count_years["Hypertension"] = len(df[df["ucodint"].isin([401, 403])])*scale
          count_years["Chronic respiratory diseases"] = len(df[df["ucodint"].isin(list(range(490, 497)))])*scale
          count_years["Ulcer"] = len(df[df["ucodint"].isin(list(range(531, 534)))])*scale
          count_years["Suicide"] = len(df[df["ucodint"].isin(list(range(950, 960)))])*scale
          count_years["Homicide"] = len(df[df["ucodint"].isin(list(range(960, 979)))])*scale
        elif file_year >= 1999:
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
          count_years["Motor vehicle accidents"] = len(df[((df["ucodint"] >= ICD.toint("V02")) & (df["ucodint"] <= ICD.toint("V04"))) | ((df["ucodint"] >= ICD.toint("V12")) & (df["ucodint"] <= ICD.toint("V14"))) | ((df["ucodfloat"] >= ICD.tofloat("V19.0")) & (df["ucodfloat"] < ICD.tofloat("V19.3"))) | ((df["ucodfloat"] > ICD.tofloat("V19.3")) & (df["ucodfloat"] < ICD.tofloat("V19.7"))) | ((df["ucodint"] >= ICD.toint("V20")) & (df["ucodint"] <= ICD.toint("V79"))) | ((df["ucodfloat"] > ICD.tofloat("V80.2")) & (df["ucodfloat"] < ICD.tofloat("V80.6"))) | ((df["ucodfloat"] >= ICD.tofloat("V81.0")) & (df["ucodfloat"] < ICD.tofloat("V81.2"))) | ((df["ucodfloat"] >= ICD.tofloat("V82.0")) & (df["ucodfloat"] < ICD.tofloat("V82.2"))) | ((df["ucodint"] >= ICD.toint("V83")) & (df["ucodint"] <= ICD.toint("V86"))) | ((df["ucodfloat"] >= ICD.tofloat("V87.0")) & (df["ucodfloat"] < ICD.tofloat("V87.9"))) | ((df["ucodfloat"] >= ICD.tofloat("V88.0")) & (df["ucodfloat"] < ICD.tofloat("V88.9"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V09.0"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V89.0"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V09.2"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V89.2")))])*scale
          count_years["Accidents excluding motor vehicles"] = len(df[((df["ucodint"] >= ICD.toint("V05")) & (df["ucodint"] <= ICD.toint("V08"))) | ((df["ucodfloat"] > ICD.tofloat("V09.2")) & (df["ucodint"] <= ICD.toint("V11"))) | ((df["ucodint"] >= ICD.toint("V15")) & (df["ucodint"] <= ICD.toint("V18"))) | ((df["ucodfloat"] > ICD.tofloat("V19.6")) & (df["ucodfloat"] < ICD.tofloat("V20"))) | ((df["ucodfloat"] >= ICD.tofloat("V80.0")) & (df["ucodfloat"] < ICD.tofloat("V80.3"))) | ((df["ucodfloat"] > ICD.tofloat("V80.5")) & (df["ucodfloat"] < ICD.tofloat("V81"))) | ((df["ucodfloat"] > ICD.tofloat("V81.1")) & (df["ucodfloat"] < ICD.tofloat("V82"))) | ((df["ucodfloat"] > ICD.tofloat("V82.1")) & (df["ucodfloat"] < ICD.tofloat("V83"))) | ((df["ucodfloat"] > ICD.tofloat("V89.2")) & (df["ucodint"] <= ICD.toint("X59"))) | ((df["ucodint"] >= ICD.toint("Y85")) & (df["ucodint"] <= ICD.toint("Y86"))) | (df["ucodint"] == ICD.toint("V01")) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V09.1"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V19.3"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V87.9"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V88.9"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("V89.1")))])*scale
          count_years["Typhoid fever"] = len(df[numpy.isclose(df["ucodfloat"], ICD.tofloat("A01.0"))])*scale
          count_years["Measles"] = len(df[df["ucodint"] == ICD.toint("B05")])*scale
          count_years["Whooping cough"] = len(df[df["ucodint"] == ICD.toint("A37")])*scale
          count_years["Diphtheria"] = len(df[df["ucodint"] == ICD.toint("A36")])*scale
          count_years["Intestinal infections"] = len(df[df["ucodint"].isin(list(range(ICD.toint("A07"), ICD.toint("A10"))) + [ICD.toint("A04")])])*scale
          count_years["Meningococcal infections"] = len(df[df["ucodint"] == ICD.toint("A39")])*scale
          count_years["Acute poliomyelitis"] = len(df[df["ucodint"] == ICD.toint("A80")])*scale
          count_years["Syphilis"] = len(df[df["ucodint"].isin(list(range(ICD.toint("A50"), ICD.toint("A54"))))])*scale
          count_years["Acute rheumatic fever"] = len(df[df["ucodint"].isin(list(range(ICD.toint("I00"), ICD.toint("I03"))))])*scale
          count_years["Hypertension"] = len(df[df["ucodint"].isin(list(range(ICD.toint("I10"), ICD.toint("I13"))))])*scale
          count_years["Chronic respiratory diseases"] = len(df[df["ucodint"].isin(list(range(ICD.toint("J40"), ICD.toint("J48"))) + [ICD.toint("J67")])])*scale
          count_years["Ulcer"] = len(df[df["ucodint"].isin(list(range(ICD.toint("K25"), ICD.toint("K29"))))])*scale
          count_years["Suicide"] = len(df[((df["ucodint"] >= ICD.toint("X60")) & (df["ucodint"] <= ICD.toint("X84"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("Y87.0")))])*scale
          count_years["Homicide"] = len(df[((df["ucodint"] >= ICD.toint("X85")) & (df["ucodint"] <= ICD.toint("Y09"))) | (df["ucodint"] == ICD.toint("Y35")) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("Y87.1"))) | (numpy.isclose(df["ucodfloat"], ICD.tofloat("Y89.0")))])*scale

    df = pandas.DataFrame.from_dict(counts, orient="index")
    output_file = os.path.abspath("data/ucod/united_states/comparable_data_since_1959.xlsx")
    df.to_excel(output_file)
    print("Created {}".format(output_file))

  def create_comparable(self):
    comparability_ratios = pandas.read_excel(self.options.comparable_ratios_input_file, sheet_name="Comparability Ratios", index_col=0, usecols=[0, 2, 4, 6, 8, 10]).fillna(1)
    comparable_ucods = pandas.read_excel(self.options.comparable_ratios_input_file, index_col="Year")
    comparable_ucods = comparable_ucods.transform(self.transform_row, axis="columns", comparability_ratios=comparability_ratios)
    comparable_ucods.to_excel(self.options.comparable_ratios_output_file)
    print("Created {}".format(self.options.comparable_ratios_output_file))
    
  def transform_row(self, row, comparability_ratios):
    icd = row["ICD Revision"]
    currenticd = 10
    if icd < currenticd:
      icd_index = int(icd - 5) # Data starts at 5th ICD
      for column in row.index.values:
        if column in comparability_ratios.index:
          ratios = comparability_ratios.loc[column]
          ratios = ratios.iloc[icd_index:]
          row[column] = row[column] * numpy.prod(ratios.values)
    return row

  def create_with_multi_index2(self, d, indexcols):
    if len(d) > 0:
      reform = {(firstKey, secondKey): values for firstKey, secondDict in d.items() for secondKey, values in secondDict.items()}
      return pandas.DataFrame.from_dict(reform, orient="index").rename_axis(indexcols).sort_index()
    else:
      return None

  def print_processing_csv(self, i, csv, csvs):
    print("Processing {} ({} of {})".format(csv, i+1, len(csvs)))
    
  def icd_query(self, codes):
    result = ""
    atol = 1e-08
    codes_pieces = codes.split(",")
    for codes_piece in codes_pieces:
      if len(result) > 0:
        result += " | "
      result += "("
      codes_piece = codes_piece.strip()
      if "-" in codes_piece:
        range_pieces = codes_piece.split("-")
        x = range_pieces[0]
        y = range_pieces[1]
        if "." in x:
          floatval = vbp.ucod.icd.ICD.tofloat(x)
          result += "(ucodfloat >= {})".format(floatval - atol)
        else:
          result += "(ucodint >= {})".format(vbp.ucod.icd.ICD.toint(x))
        result += " & "
        if "." in y:
          floatval = vbp.ucod.icd.ICD.tofloat(y)
          result += "(ucodfloat <= {})".format(floatval + atol)
        else:
          result += "(ucodint <= {})".format(vbp.ucod.icd.ICD.toint(y))
      else:
        if "." in codes_piece:
          floatval = vbp.ucod.icd.ICD.tofloat(codes_piece)
          result += "(ucodfloat >= {}) & (ucodfloat <= {})".format(floatval - atol, floatval + atol)
        else:
          result += "ucodint == {}".format(vbp.ucod.icd.ICD.toint(codes_piece))
      result += ")"

    return result

  def get_calculated_scale_function_values(self):
    self.check_raw_files_directory()
    average_range = self.options.average_age_range
    min_year = self.data["Year"].max() - self.options.average_age_range + 1
    
    if self.options.data_type == DataType.UCOD_1999_2017_SUB_CHAPTERS:
      icd_codes = self.data["ICD Sub-Chapter Code"].unique()
      icd_codes_map = dict(zip(icd_codes, [{} for i in range(0, len(icd_codes))]))
    elif self.options.data_type == DataType.UCOD_LONGTERM_COMPARABLE_LEADING:
      ulcl_codes = pandas.read_excel(
        self.options.comparable_ratios_input_file,
        index_col=0,
        sheet_name="Comparability Ratios",
        usecols=[0, 11],
        squeeze=True,
      )
      icd_codes = ulcl_codes.unique()
      icd_codes_map = dict(zip(icd_codes, [{} for i in range(0, len(icd_codes))]))
    else:
      raise NotImplementedError()

    csvs = self.get_mortality_files()
    stats = {}
    max_year = sys.maxsize
    for i, csv in enumerate(csvs):
      filename, file_extension, file_year = self.get_mortality_file_info(csv)
      if filename and file_year >= min_year and file_year <= max_year:
        self.print_processing_csv(i, csv, csvs)
        year_stats = {}
        stats[file_year] = year_stats
        df, scale = self.get_mortality_data(csv, file_year)
        for icd_range, trash in icd_codes_map.items():
          ages = df.query(self.icd_query(icd_range))["AgeMinutes"]
          year_stats[icd_range] = {"Sum": ages.sum(), "Max": ages.max(), "Count": ages.count(), "Scale": scale}
    
    codescol = "Codes"
    statsdf = self.create_with_multi_index2(stats, ["Year", codescol])
    statsdf = statsdf.dropna()
    self.write_spreadsheet(statsdf, self.prefix_all("statsdf"))
    subset = statsdf.loc[statsdf.index.max()[0]-average_range:statsdf.index.max()[0]]
    deathmax = statsdf["Max"].max()
    calculated_col = self.options.average_ages
    agesbygroup = subset.groupby(codescol).apply(lambda row: 1 - ((row["Sum"].sum() / row["Count"].sum()) / deathmax)).sort_values().rename(calculated_col).to_frame()
    agesbygroup["MaxAgeMinutes"] = deathmax
    agesbygroup["MaxAgeYears"] = agesbygroup["MaxAgeMinutes"] / 525960
    agesbygroup["AverageAgeMinutes"] = subset.groupby(codescol).apply(lambda row: row["Sum"].sum() / row["Count"].sum())
    agesbygroup["AverageAgeYears"] = agesbygroup["AverageAgeMinutes"] / 525960
    agesbygroup["SumAgeMinutes"] = subset.groupby(codescol).apply(lambda row: row["Sum"].sum())
    agesbygroup["SumAgeYears"] = agesbygroup["SumAgeMinutes"] / 525960
    agesbygroup["Count"] = subset.groupby(codescol).apply(lambda row: row["Count"].sum())

    if self.options.data_type == DataType.UCOD_1999_2017_SUB_CHAPTERS:
      agesbygroup[self.obfuscated_column_name] = agesbygroup.apply(lambda row: self.get_obfuscated_name(self.data[self.data["ICD Sub-Chapter Code"] == row.name][self.get_action_column_name()].iloc[0]), raw=True, axis="columns")
    elif self.options.data_type == DataType.UCOD_LONGTERM_COMPARABLE_LEADING:
      agesbygroup[self.obfuscated_column_name] = agesbygroup.apply(lambda row: self.get_obfuscated_name(ulcl_codes[ulcl_codes == row.name].index.values[0]), raw=True, axis="columns")
    else:
      raise NotImplementedError()

    self.write_spreadsheet(agesbygroup, self.prefix_all("agesbygroup"))
    result = agesbygroup[[self.obfuscated_column_name, calculated_col]]
    result.set_index(self.obfuscated_column_name, inplace=True)
    #result[self.options.average_ages] = vbp.normalize(result[self.options.average_ages].values, 0.5, 1.0)
    return result
  
  def run_test(self):
    self.ensure_loaded()
    return None
