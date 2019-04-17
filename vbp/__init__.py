"""
Value Based Prioritization (VBP) uses value theory to quantitatively
prioritize potential actions to accomplish a goal:

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
"""

import os
import abc
import sys
import enum
import math
import numpy
import pandas
import random
import shutil
import logging
import pathlib
import argparse
import datetime
import warnings
import fbprophet
import traceback
import matplotlib
import collections
import matplotlib.offsetbox
import statsmodels.tools
import statsmodels.formula.api

VERSION = "0.3.4"
numpy.seterr("raise")

def linear_regression_formula(degree=1):
  return "y ~ " + " + ".join("I(x**{})".format(i) for i in range(1, degree+1))

def linear_regression(df, x, y, formula):
  data = {"x": df[x].values, "y": df[y].values}

  # https://www.statsmodels.org/dev/regression.html
  # https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html
  # https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLSResults.html
  ols_result = statsmodels.formula.api.ols(formula, data).fit()

  return ols_result

def linear_regression_modeled_formula(ols_result, degree=1, truncate_scientific=False):
  if truncate_scientific:
    return "y ~ " + " + ".join("({:.2E}*x{})".format(ols_result.params[i], "^" + str(i) if i > 1 else "") for i in range(1, degree+1)) + " + {:.2E}".format(ols_result.params[0])
  else:
    return "y ~ " + " + ".join("({}*x{})".format(ols_result.params[i], "^" + str(i) if i > 1 else "") for i in range(1, degree+1)) + " + {}".format(ols_result.params[0])

# https://en.wikipedia.org/wiki/Normalization_(statistics)
def normalize(nparray, min, max):
  return numpy.vectorize(normalize_func(nparray, min, max))(nparray)

def normalize_func(nparray, min, max):
  # (((element - min(nparray)) * (max - min)) / (max(nparray) - min(nparray))) + min
  min_nparray = nparray.min()
  max_nparray = nparray.max()
  return normalize_func_explicit_minmax(nparray, min_nparray, max_nparray, min, max)

def normalize_func_explicit_minmax(nparray, min_range, max_range, min, max):
  # (((element - min_range) * (max - min)) / (max_range - min_range)) + min
  minmax_range_diff = max_range - min_range
  minmaxdiff = max - min
  # (((element - min_range) * minmaxdiff) / minmax_range_diff) + min
  # ((element * minmaxdiff) - (min_range * minmaxdiff)) / minmax_range_diff) + min
  min_range_minmaxdiff = min_range * minmaxdiff
  return lambda x: (((x * minmaxdiff) - min_range_minmaxdiff) / minmax_range_diff) + min

def normalize0to1(nparray):
  return normalize(nparray, 0, 1)

class DetailedErrorArgumentParser(argparse.ArgumentParser):
  def error(self, message):
    sys.stderr.write("error: {}\n\n".format(message))
    self.print_help()
    sys.exit(1)

def create_parser():
  return DetailedErrorArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def print_full_columns(df):
  with pandas.option_context("display.max_columns", None):
    with pandas.option_context("display.max_colwidth", sys.maxsize):
      print(df)

def print_all_rows(df):
  with pandas.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)

class DataSourceDataType(enum.Enum):
  def __str__(self):
    return self.name
  
  @classmethod
  def _missing_(cls, name):
    for member in cls:
      if member.name.lower() == name.lower():
        return member
  
class DataSource(abc.ABC):
  """The DataSource class is the base abstract class for VBP."""
  
  ##############
  # Attributes #
  ##############
  pretty_action_column_name = "Action"
  action_number_column_name = "ActionNumber"
  obfuscated_column_name = "Name"
  predict_column_name = "Predicted"
  obfuscated_action_names = {}
  obfuscated_action_names_count = 0
  available_obfuscated_names = None
  model_fit_algorithms = {"lowest_aic": "AIC", "lowest_aicc": "AICc", "lowest_bic": "BIC"}
  default_model_fit_algorithm = "lowest_aicc"
  default_cache_dir = "vbpcache"
  default_output_dir = "vbpoutput"

  #######################
  # Main Public Methods #
  #######################
  def load(self, args):
    self.prepare(args)
    return self
  
  def modeled_value_based_prioritization(self):
    return self.run_modeled_value_based_prioritization()
  
  def predict(self):
    return self.run_predict()

  def get_possible_actions(self):
    return self.run_get_possible_actions()
  
  def get_action_data(self, action, interpolate=True):
    return self.run_get_action_data(action, interpolate)
  
  def prepare_data(self):
    return self.run_prepare_data()

  def test(self):
    return self.run_test()
  
  @staticmethod
  def get_data_types_enum():
    return None

  @staticmethod
  def get_data_types_enum_default():
    return None
  
  ####################
  # Abstract Methods #
  ####################
  @abc.abstractmethod
  def run_load(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_action_column_name(self):
    raise NotImplementedError()
  
  @abc.abstractmethod
  def get_value_column_name(self):
    raise NotImplementedError()
  
  @abc.abstractmethod
  def run_predict(self):
    raise NotImplementedError()

  #################
  # Other Methods #
  #################
  def initialize_parser(self, parser):
    return None

  def prepare(self, args):
    self.ensure_options(args)
    self.ensure_loaded()

  def get_data_dir(self):
    if os.path.isdir("vbp"):
      return "vbp"
    else:
      return os.path.dirname(__file__)

  def ensure_options(self, args):
    if not hasattr(self, "options"):
      parser = create_parser()
      self.initialize_parser(parser)
      parser.add_argument("-b", "--best-fit", choices=list(self.model_fit_algorithms.keys()), default=self.default_model_fit_algorithm, help="Best fitting model algorithm")
      parser.add_argument("--cachedir", help="cache directory", default=self.default_cache_dir)
      parser.add_argument("--clean", action="store_true", help="first clean any existing generated data such as images", default=False)
      parser.add_argument("-d", "--hide-graphs", dest="show_graphs", action="store_false", help="hide graphs")
      data_type_choices = self.get_data_types_enum()
      if data_type_choices is None:
        data_type_choices = []
      else:
        data_type_choices = list(data_type_choices)
      parser.add_argument(
        "--data-type",
        help="The type of data to process",
        type=self.get_data_types_enum() if self.get_data_types_enum() is not None else str,
        default=self.get_data_types_enum_default(),
        choices=data_type_choices,
      )
      parser.add_argument("--debug", action="store_true", help="Debug", default=False)
      parser.add_argument("--do-not-clean", action="store_true", help="Do not clean output data", default=False)
      parser.add_argument("--do-not-exit-on-warning", dest="exit_on_warning", action="store_false", help="Do not exit on a warning")
      parser.add_argument("--do-not-obfuscate", action="store_true", help="do not obfuscate action names", default=False)
      parser.add_argument("--do-not-write-spreadsheets", dest="write_spreadsheets", action="store_false", help="do not write spreadsheets")
      parser.add_argument("-k", "--top-actions", help="Number of top actions to report", type=int, default=5)
      parser.add_argument("--manual-scales", help="manually calculated scale functions")
      parser.add_argument("--max-title-length", help="Maximum title length, particular for charts", type=int, default=50)
      parser.add_argument("--no-data-type-subdir", help="Do not create an output subdirectory based on the data type", dest="data_type_subdir", action="store_false")
      parser.add_argument("-o", "--output-dir", help="output directory", default=self.default_output_dir)
      parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=10)
      parser.add_argument("-s", "--show-graphs", dest="show_graphs", action="store_true", help="verbose")
      parser.add_argument("-v", "--verbose", action="store_true", help="verbose", default=False)
      parser.set_defaults(
        show_graphs=False,
        data_type_subdir=True,
        exit_on_warning=True,
        write_spreadsheets=True,
      )
      self.options = parser.parse_args(args)
      
      clean = self.options.clean
      if self.options.output_dir == self.default_output_dir:
        # Always clean if it's the default output directory
        # Unless --do-not-clean is explicitly used
        clean = True

      if not self.options.do_not_clean and clean and os.path.exists(self.options.output_dir):
        self.clean_files()
        
      if not os.path.exists(self.options.output_dir):
        os.makedirs(self.options.output_dir)

  def ensure_loaded(self):
    if not hasattr(self, "data"):
      self.data = self.run_load()
      self.post_process()
      
  def check_action_exists(self, action):
    if action not in self.get_possible_actions():
      raise ValueError("Could not find action: {}".format(action))
  
  def post_process(self):
    self.data[self.obfuscated_column_name] = self.data[self.get_action_column_name()].apply(lambda x: self.get_obfuscated_name(x))
    self.write_spreadsheet(self.data, self.prefix_all("data_processed"))

  def run_modeled_value_based_prioritization(self):
    manual_scales = None
    
    if self.options.manual_scales is not None:
      manual_scales_file = self.get_manual_scales_file(self.options.manual_scales)
      if not os.path.exists(manual_scales_file):
        manual_scales_file = self.options.manual_scales
      
      if manual_scales_file.endswith("xlsx"):
        manual_scales = pandas.read_excel(manual_scales_file)
      elif manual_scales_file.endswith("csv"):
        manual_scales = pandas.read_csv(manual_scales_file)
      else:
        raise ValueError("Could not infer type of manually calculated scale function file (expecting extension .xlsx or .csv)")
      
      manual_scales[self.obfuscated_column_name] = manual_scales[self.obfuscated_column_name].apply(lambda x: self.get_obfuscated_name(x))
      manual_scales.set_index(self.obfuscated_column_name, inplace=True)
      manual_scales.drop(columns=[self.pretty_action_column_name, self.action_number_column_name], inplace=True, errors="ignore")
    
    b = self.predict()
    
    predict_sum = b[self.predict_column_name].sum()
    b[self.predict_column_name] = b[self.predict_column_name].divide(predict_sum)

    if manual_scales is not None:
      b = b.join(manual_scales)
    
    calculated_scales = self.get_calculated_scale_function_values()
    if calculated_scales is not None and calculated_scales.empty is False:
      b = b.join(calculated_scales)
    
    b = b.reindex(columns=sorted(b.columns))
    
    self.write_spreadsheet(b, self.prefix_all("b2"))
    
    z = b.prod(axis="columns").sort_values(ascending=False)
    z.name = "Z({})".format(self.options.predict)
    z = z.to_frame()
    z.reset_index(inplace=True)
    z.index.name = "k"
    z.index += 1
    
    self.write_spreadsheet(z, self.prefix_all("z"))
    
    z = z.head(self.options.top_actions)

    return z

  def run_get_possible_actions(self):
    df = self.data[self.get_action_column_name()]
    # Randomize the list to reduce bias
    df = df.sample(frac=1)
    df = df.unique()
    return df

  def get_obfuscated_name(self, s):
    if self.options.do_not_obfuscate:
      return s
    result = self.obfuscated_action_names.get(s)
    if result is None:
      if self.available_obfuscated_names is None:
        self.obfuscated_action_names_count = len(self.get_possible_actions())
        self.available_obfuscated_names = list(range(1, self.obfuscated_action_names_count + 1))
        random.shuffle(self.available_obfuscated_names)
      
      if len(self.available_obfuscated_names) == 0:
        print(self.obfuscated_action_names)
        raise ValueError("Tried to obfuscate too many items. This one: {}".format(s))
      format_string = "{}{:0" + str(len(str(self.obfuscated_action_names_count))) + "d}"
      result = format_string.format(self.obfuscated_column_name, self.available_obfuscated_names.pop())
      
      self.obfuscated_action_names[s] = result
    return result

  def clean_files(self):
    for e in os.listdir(self.options.output_dir):
      p = os.path.join(self.options.output_dir, e)
      if os.path.isfile(p):
        os.unlink(p)
      else:
        shutil.rmtree(p)

  def create_output_name(self, name):
    name = name.replace(" ", "_") \
               .replace("(", "") \
               .replace(")", "") \
               .replace("[", "") \
               .replace("]", "") \
               .replace("{", "") \
               .replace("}", "") \
               .replace("/", "_") \
               .replace(":", "_") \
               .replace("\\", "_") \
               .replace("<", "_") \
               .replace(">", "_") \
               .replace("!", "_") \
               .replace("\"", "_") \
               .replace("'", "_") \
               .replace("|", "_") \
               .replace("?", "_") \
               .replace(",", "") \
               .replace("*", "_")

    # Remove any periods from the part before an extension
    if len(name) > 4 and name[len(name)-4] == "." and name.find(".") < (len(name) - 4):
      name = name[:name.rfind(".")].replace(".", "_") + name[name.rfind("."):]

    outputdir = self.options.output_dir
    if self.options.data_type is not None and self.options.data_type_subdir:
      outputdir = os.path.join(outputdir, self.options.data_type.name)
      if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    return os.path.join(outputdir, name)
  
  def write_spreadsheet(self, df, name, use_digits_grouping_number_format=False):
    if self.options.write_spreadsheets:
      df.to_csv(self.create_output_name(name + ".csv"))

      df.to_pickle(self.create_output_name(name + ".pkl"))
      
      writer = pandas.ExcelWriter(self.create_output_name(name + ".xlsx"), engine="xlsxwriter")
      
      sheet_name = "Sheet1"
      df.to_excel(writer, sheet_name=sheet_name)
      
      digits_grouping_format = None
      if use_digits_grouping_number_format:
        digits_grouping_format = writer.book.add_format({"num_format": "#,##0.00"})

      # Set column widths based on longest values
      # https://stackoverflow.com/a/40535454/11030758
      worksheet = writer.sheets[sheet_name]
      df2 = df.reset_index(level=df.index.names)
      for idx, col in enumerate(df2):  # loop through all columns
        series = df2[col]
        max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                  )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len, digits_grouping_format)  # set column width
      
      writer.save()
  
  def write_verbose(self, file, obj):
    s = str(obj)
    self.write(file, s)
    if self.options.verbose:
      print(s)

  def write(self, file, str):
    with open(self.create_output_name(file), "w") as f:
      f.write(str)

  def find_best_fitting_models(self, df):
    result = df.groupby(self.obfuscated_column_name).apply(lambda x: self.best_fitting_model(x))
    
    # Drop all but a single Action index. Some other indices may
    # accumulate as part of the grouping and averaging (variable
    # depending on pandas optimizations).
    while result.index.nlevels > 1:
      action_index = result.index.names.index(self.obfuscated_column_name)
      if action_index == 0:
        result.index = result.index.droplevel(1)
      else:
        result.index = result.index.droplevel(0)
    
    # Update any negative predictions to 0
    result[self.predict_column_name] = result[self.predict_column_name].apply(lambda x: 0 if x < 0 else x)
    
    return result
      
  def best_fitting_model_grouped(self, df, algorithm):
    if self.options.verbose:
      print("=================\nGrouped Incoming:\n{}\n".format(df))
      
    result = getattr(self, "best_fitting_model_{}".format(algorithm))(df)
    if result.empty:
      raise ValueError("Empty best fit model for {}".format(result))
    elif len(result) > 1:
      # Multiple results, so just print a message and pick the first one
      self.print_info("More than one model matches {}:\n{}".format(algorithm, result))
      result = result.iloc[[0]]

    if self.options.verbose:
      print("Grouped Outgoing:\n{}\n=================".format(result))
    
    return result
      
  def best_fitting_model(self, df):
    if self.options.verbose:
      print("=========\nIncoming:\n{}\n".format(df))
      
    # For each ModelType, select the best model. We end up
    # with a single row for each ModelType. We start with
    # the selected (or default) fit algorithm, but if the result
    # is infinity, then we move onto the next one.
    algorithms = list(self.model_fit_algorithms.keys()).copy()
    algorithms.remove(self.options.best_fit)
    algorithms.insert(0, self.options.best_fit)
    for algorithm in algorithms:
      result = df.groupby("ModelType").apply(lambda x: self.best_fitting_model_grouped(x, algorithm))
      if not numpy.isinf(result[self.model_fit_algorithms[algorithm]][0]):
        break

    if len(result) > 1:
      # Finally we take the averages across model groups
      result = result.groupby(self.obfuscated_column_name).mean()
      
    if self.options.verbose:
      print("Outgoing:\n{}\n=========".format(result))
      
    return result
  
  def best_fitting_model_lowest_aicc(self, df):
    return df[df.AICc == df.AICc.min()]
  
  def best_fitting_model_lowest_aic(self, df):
    return df[df.AIC == df.AIC.min()]
  
  def best_fitting_model_lowest_bic(self, df):
    return df[df.BIC == df.BIC.min()]

  def print_warning(self, message):
    print("WARNING: {}".format(message))
    if self.options.exit_on_warning:
      sys.exit(1)

  def print_info(self, message):
    print("INFO: {}".format(message))

  def prefix_all(self, name):
    return "_all_" + name

  def create_model_prophet(self, action, i, count):
    print("Creating Prophet model {} of {} for: {}".format(i, count, self.get_obfuscated_name(action)))
    df = self.get_action_data(action, interpolate=False)
    
    model_map = self.get_model_map_base()

    df = df[[self.get_value_column_name()]]
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "ds", self.get_value_column_name(): "y"}, inplace=True)
    # https://facebook.github.io/prophet/docs/saturating_forecasts.html
    df["floor"] = 0

    prophet = fbprophet.Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    prophet.fit(df, algorithm="Newton") # https://github.com/facebook/prophet/issues/870
    future = prophet.make_future_dataframe(periods=self.options.predict, freq="Y")
    forecast = prophet.predict(future)
    
    prophet.plot(forecast, xlabel="Date", ylabel=self.get_value_column_name())
    self.save_plot_image(action, "forecast")
    
    prophet.plot_components(forecast)
    self.save_plot_image(action, "components")

    self.write_spreadsheet(forecast, "{}_prophetforecast".format(self.get_obfuscated_name(action)))
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    
    index = (self.get_obfuscated_name(action), "Prophet")
    model_map["ModelType"][index] = "Prophet"
    model_map[self.predict_column_name + "Year"][index] = forecast.iloc[-1]["ds"].year
    model_map[self.predict_column_name][index] = forecast.iloc[-1]["yhat"]
    
    lasttwopoints = forecast.iloc[-2::,]
    slope = (lasttwopoints.iloc[1,]["yhat"] - lasttwopoints.iloc[0,]["yhat"]) / (lasttwopoints.iloc[1,]["ds"].year - lasttwopoints.iloc[0,]["ds"].year)
    
    model_map[self.predict_column_name + "Derivative"][index] = slope
    model_map["AIC"][index] = 0
    model_map["AICc"][index] = 0
    model_map["BIC"][index] = 0
    model_map["SSE"][index] = 0
      
    resultdf = pandas.DataFrame(model_map)
    resultdf.index.rename([self.obfuscated_column_name, "Model"], inplace=True)

    self.write_spreadsheet(df, "{}_prophetdata".format(self.get_obfuscated_name(action)))
    self.write_spreadsheet(resultdf, "{}_prophetresults".format(self.get_obfuscated_name(action)))

    return resultdf

  def run_prepare_data(self):
    return False

  def run_test(self):
    return False

  def get_calculated_scale_function_values(self):
    return None

  def get_enum_names(self, e):
    return list(map(lambda x: x.name, list(e)))

  def run_get_action_data(self, action, interpolate):
    return self.data[self.data[self.get_action_column_name()] == action]

  def save_plot_image(self, action, context, fig=None, legend=False):
    
    if legend:
      # https://stackoverflow.com/questions/54791323/
      matplotlib.pyplot.legend()
    
    #fig.set_size_inches(10, 5)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(self.create_output_name("{}_{}.png".format(self.get_obfuscated_name(action), context)), dpi=100)
    
    if self.options.show_graphs:
      matplotlib.pyplot.show()

    if fig is not None:
      matplotlib.pyplot.close(fig)
      
  def get_manual_scales_file(self, filename):
    nameaddition = ""
    datatype = self.options.data_type
    if datatype is None:
      datatype = self.get_data_types_enum_default()
      
    if datatype is not None:
      nameaddition = "_{}".format(datatype.name)
      
    outputname = filename
    outputname = outputname[:outputname.rindex(".")] + nameaddition + outputname[outputname.rindex("."):]
    return outputname

  def set_or_append(self, df, append):
    if df is None:
      return append
    else:
      if append is None:
        return df
      else:
        return pandas.concat([df, append], sort=False)

  def check_cache(self, name):
    if not os.path.exists(self.options.cachedir):
      os.makedirs(self.options.cachedir)
    path = os.path.join(self.options.cachedir, name + ".pkl")
    if os.path.exists(path):
      print("Reading cached data for {} from {}".format(name, path))
      return pandas.read_pickle(path)
    else:
      return None
  
  def write_cache(self, name, df):
    if not os.path.exists(self.options.cachedir):
      os.makedirs(self.options.cachedir)
    path = os.path.join(self.options.cachedir, name + ".pkl")
    df.to_pickle(path)
    print("Cached {} to {}".format(name, path))
    
  def load_with_cache(self, name, loadfunc, *args):
    result = self.check_cache(name)
    if result is None:
      result = loadfunc(*args)
      self.write_cache(name, result)
    return result
  
  def get_chart_title(self, title):
    max_len = self.options.max_title_length
    if len(title) > max_len:
      # First save off anything in parentheses:
      after = ""
      if "(" in title:
        after = " " + title[title.find("("):]
        title = title[:title.find("(")].strip()
        max_len = max_len - len(after) + 1
        
      title = title[:max_len] + "..." + after

    return title

class TimeSeriesDataSource(DataSource):
  def initialize_parser(self, parser):
    parser.add_argument("action", nargs="*", help="Action name")
    parser.add_argument("--base-column", help="Name of the base column", default="S1")
    parser.add_argument("--derivative-column", help="Name of the derivative column", default="SD")
    parser.add_argument("--ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_true")
    parser.add_argument("--ets-no-additive-models", help="Do not use additive models", action="store_true", default=False)
    parser.add_argument("--ets-no-multiplicative-models", help="Do not use multiplicative models", action="store_true", default=False)
    parser.add_argument("--no-ets", help="Exponential smoothing using Holt's linear trend method", dest="ets", action="store_false")
    parser.add_argument("--ols", help="Ordinary least squares", dest="ols", action="store_true")
    parser.add_argument("--no-ols", help="Ordinary least squares", dest="ols", action="store_false")
    parser.add_argument("--ols-min-degrees", help="minimum polynomial degree", type=int, default=1)
    parser.add_argument("--ols-max-degrees", help="maximum polynomial degree", type=int, default=1)
    parser.add_argument("--prophet", help="Run prophet algorithm", dest="prophet", action="store_true")
    parser.add_argument("--no-prophet", help="Do not run the prophet algorithm", dest="prophet", action="store_false")
    parser.set_defaults(
      ets=True,
      ols=False,
      prophet=False,
    )

  def get_actions(self):
    actions = self.options.action
    if actions is None or len(actions) == 0:
      actions = self.get_possible_actions()
    return actions

  def run_get_action_data(self, action, interpolate):
    df = self.data[self.data[self.get_action_column_name()] == action]
    if df.index.name != "Date":
      df = df.set_index("Date")
    if interpolate:
      # ETS requires a specific frequency so we forward-fill any missing
      # years. If there's already an annual frequency, no change is made
      # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
      df = df.resample("AS").ffill().dropna()
    if "Year" not in df.columns:
      df["Year"] = df.index.map(lambda d: d.year)
    return df
  
  def run_predict(self):
    actions = self.get_actions()
      
    count = len(actions)
    if self.options.verbose:
      print("{} actions".format(count))
    
    model_results = None
    
    action_count = 1
    for action in actions:
      self.check_action_exists(action)
      try:
        if self.options.ols:
          for i in range(self.options.ols_min_degrees, self.options.ols_max_degrees + 1):
            model_results = self.set_or_append(model_results, self.create_model_ols(action, i, self.options.predict, action_count, count))
        if self.options.ets:
          model_results = self.set_or_append(model_results, self.create_model_ets(action, self.options.predict, action_count, count))
        if self.options.prophet:
          model_results = self.set_or_append(model_results, self.create_model_prophet(action, action_count, count))
      except:
        if self.options.verbose:
          print(action)
        traceback.print_exc()
        break
      
      action_count += 1
      
    if model_results is not None and len(model_results) > 0:
      
      self.write_spreadsheet(model_results, self.prefix_all("model_results"))
      
      m = self.find_best_fitting_models(model_results)
      
      m[self.options.base_column] = 1.0
      
      if len(m) > 1:
        m[self.options.derivative_column] = normalize(m.PredictedDerivative.values, 0.5, 1.0)
      else:
        m[self.options.derivative_column] = 1.0
      
      self.write_spreadsheet(m, self.prefix_all("m"))

      extra_columns = [self.options.base_column, self.options.derivative_column]
      final_columns = [self.predict_column_name]

      extra_columns.sort()
      final_columns += extra_columns
      b = m[final_columns]
      
      self.write_spreadsheet(b, self.prefix_all("b1"))
      return b

    else:
      return None
  
  def get_action_title_prefix(self):
    return ""
  
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
    df = self.get_action_data(action)
    
    df_nozeros = df[df[self.get_value_column_name()] != 0]

    if len(df) <= 1 or len(df_nozeros) == 0:
      return None

    # First create an image with raw data and no predictions because
    # an exponential prediction may make it difficult to see the
    # raw data trend
    
    action_title = "{}{}".format(self.get_action_title_prefix(), self.get_obfuscated_name(action))
    
    self.write_spreadsheet(df, "{}_etsdata".format(self.get_obfuscated_name(action)))

    df = df[[self.get_value_column_name()]]
    ax = df.plot(color="black", marker="o", legend=True, title=self.get_chart_title(action_title), grid=True, kind="line")
    ax.set_ylabel(self.get_value_column_name())
    fig = matplotlib.pyplot.gcf()
    self.save_plot_image(action, "ets_nopredictions", fig, True)
    
    # Now build the plot with everything and the predictions
    df = df[[self.get_value_column_name()]]
    ax = df.plot(color="black", marker="o", legend=True, title=self.get_chart_title(action_title), grid=True, kind="line")
    ax.set_ylabel(self.get_value_column_name())
    fig = matplotlib.pyplot.gcf()
    
    model_map = self.get_model_map_base()
    
    results = {}
    if not self.options.ets_no_additive_models:
      results.update(self.run_ets(df[self.get_value_column_name()], color="red", predict=predict, exponential=False, damped=False))
      results.update(self.run_ets(df[self.get_value_column_name()], color="cyan", predict=predict, exponential=False, damped=True))
    if not self.options.ets_no_multiplicative_models:
      results.update(self.run_ets(df[self.get_value_column_name()], color="green", predict=predict, exponential=True, damped=False))
      results.update(self.run_ets(df[self.get_value_column_name()], color="blue", predict=predict, exponential=True, damped=True))
    
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
    
    self.save_plot_image(action, "ets", fig, True)
    
    for title, result in results.items():
      fig, ax = matplotlib.pyplot.subplots()
      ax.scatter(result[0].fittedvalues, result[0].resid)
      ax.set_title("Residuals of {} (${}$)".format(self.get_obfuscated_name(action), title))
      self.save_plot_image(action, "{}_ets_residuals".format(title), fig)
      
    self.write_spreadsheet(resultdf, "{}_etsresults".format(self.get_obfuscated_name(action)))
    
    return resultdf
      
  def run_ets(self, df, color, predict, exponential=False, damped=False, damping_slope=0.98):
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.Holt.html
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html
    # https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
    
    if not damped:
      damping_slope = None

    # Holt throws errors with 0 or NaN values
    df = df[df != 0].resample("AS").interpolate("linear")

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
    df = self.get_action_data(action)
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
    
    df["ScaledYear"] = numpy.vectorize(normalize_func_explicit_minmax(df.Year.values, df.Year.min(), max_year, 0, 1))(df.Year.values)
    
    dfnoNaNs = df.dropna()
    if self.options.verbose:
      print(dfnoNaNs)

    formula = linear_regression_formula(degree)
    model = linear_regression(dfnoNaNs, "ScaledYear", self.get_value_column_name(), formula)
    self.write_verbose("{}_{}.txt".format(self.get_obfuscated_name(action), degree), model.summary())
    
    index = (self.get_obfuscated_name(action), degree)
    model_map["ModelType"][index] = "OLS"
    model_map["AdjustedR2"][index] = model.rsquared_adj
    model_map["ProbFStatistic"][index] = model.f_pvalue
    model_map["LogLikelihood"][index] = model.llf
    model_map["AIC"][index] = model.aic
    model_map["BIC"][index] = model.bic
    model_map["Formula"][index] = linear_regression_modeled_formula(model, degree)
    
    model_map["AICc"][index] = statsmodels.tools.eval_measures.aicc(model.llf, model.nobs, degree + 1)
    
    predicted_values = model.fittedvalues.copy()
    actual = dfnoNaNs[self.get_value_column_name()].values.copy()
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

    self.save_plot_image(action, "{}_ols_residuals".format(degree), fig)

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

    action_title = "{}{}".format(self.get_action_title_prefix(), self.get_obfuscated_name(action))
    ax = df.plot("Year", self.get_value_column_name(), grid=True, title=self.get_chart_title(action_title), color="black", marker="o", kind="line", legend=True)

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

    self.save_plot_image(action, "{}_ols".format(degree), fig, True)

    resultdf = pandas.DataFrame(model_map)
    resultdf.index.rename([self.obfuscated_column_name, "Model"], inplace=True)

    self.write_spreadsheet(df, "{}_olsdata".format(self.get_obfuscated_name(action)))
    self.write_spreadsheet(resultdf, "{}_olsresults".format(self.get_obfuscated_name(action)))
    return resultdf

class DictTree(collections.defaultdict):
  def __init__(self, d={}, value=None, **kwargs):
    super().__init__(DictTree)
    self.update(d)
    self.update(kwargs)
    self.value = value
  
  def recursive_list(self, leaves_only):
    accumulator = []
    self.recursive_list_process(self, leaves_only, accumulator)
    return accumulator
  
  def recursive_list_process(self, node, leaves_only, accumulator):
    for k, v in node.items():
      if isinstance(v, DictTree):
        v.recursive_list_process(v, leaves_only, accumulator)
      elif isinstance(v, dict):
        self.recursive_list_process(v, leaves_only, accumulator)
      else:
        accumulator.append(v)
    if self == node and self.value is not None:
      if not leaves_only or (leaves_only and len(node) == 0):
        accumulator.append(self.value)

  def recursive_dict(self, leaves_only):
    accumulator = {}
    self.recursive_dict_process(self, leaves_only, accumulator)
    return accumulator
  
  def recursive_dict_process(self, node, leaves_only, accumulator):
    for k, v in node.items():
      if isinstance(v, DictTree):
        v.recursive_dict_process(v, leaves_only, accumulator)
        if not leaves_only and v.value is not None:
          accumulator[k] = v.value
      elif isinstance(v, dict):
        self.recursive_dict_process(v, leaves_only, accumulator)
      else:
        accumulator[k] = v

  def roots_list(self):
    accumulator = []
    for v in self.values():
      if isinstance(v, DictTree):
        accumulator.append(v.value)
      else:
        accumulator.append(v)
    return accumulator

  def find_value(self, func):
    return self.find_value_recursive(self, func)

  def find_value_recursive(self, node, func):
    for k, v in node.items():
      if self.value is not None and func(self.value):
        return self.value
      if isinstance(v, DictTree):
        result = v.find_value_recursive(v, func)
        if result is not None:
          return result
      elif isinstance(v, dict):
        result = self.find_value_recursive(v, func)
        if result is not None:
          return result
      else:
        if v is not None and func(v):
          return v
    return None
