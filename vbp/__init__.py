import os
import abc
import sys
import enum
import math
import numpy
import pandas
import random
import shutil
import pathlib
import argparse
import datetime
import matplotlib
import collections
import matplotlib.offsetbox
import statsmodels.tools
import statsmodels.formula.api

VERSION = "0.1.0"
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

class DataSourceDataType(enum.Enum):
  def __str__(self):
    return self.name
  
  @classmethod
  def _missing_(cls, name):
    for member in cls:
      if member.name.lower() == name.lower():
        return member
  
class DataSource(object, metaclass=abc.ABCMeta):
  ##############
  # Attributes #
  ##############
  pretty_action_column_name = "Action"
  action_number_column_name = "ActionNumber"
  obfuscated_column_name = "RawName"
  predict_column_name = "Predicted"
  obfuscated_action_names = {}
  obfuscated_action_names_count = 0
  available_obfuscated_names = None
  model_fit_algorithms = {"lowest_aic": "AIC", "lowest_aicc": "AICc", "lowest_bic": "BIC"}
  default_model_fit_algorithm = "lowest_aicc"

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
  
  def get_action_data(self, action):
    return self.run_get_action_data(action)
  
  def prophet(self):
    return self.run_prophet()

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
  def initialize_parser(self, parser):
    raise NotImplementedError()

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
  def prepare(self, args):
    self.ensure_options(args)
    self.ensure_loaded()

  def ensure_options(self, args):
    if not hasattr(self, "options"):
      parser = create_parser()
      self.initialize_parser(parser)
      parser.add_argument("-b", "--best-fit", choices=list(self.model_fit_algorithms.keys()), default=self.default_model_fit_algorithm, help="Best fitting model algorithm")
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
      parser.add_argument("-k", "--top-actions", help="Number of top actions to report", type=int, default=5)
      parser.add_argument("--manual-scales", help="manually calculated scale functions")
      parser.add_argument("--no-data-type-subdir", help="Do not create an output subdirectory based on the data type", dest="data_type_subdir", action="store_false")
      parser.add_argument("-o", "--output-dir", help="output directory", default="output")
      parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=10)
      parser.add_argument("-s", "--show-graphs", dest="show_graphs", action="store_true", help="verbose")
      parser.add_argument("-v", "--verbose", action="store_true", help="verbose", default=False)
      parser.set_defaults(
        show_graphs=False,
        data_type_subdir=True,
        exit_on_warning=True,
      )
      self.options = parser.parse_args(args)
      
      clean = self.options.clean
      if self.options.output_dir == "output":
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
    
    self.write_spreadsheet(b, self.prefix_all("b"))
    
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
    outputdir = self.options.output_dir
    if self.options.data_type is not None and self.options.data_type_subdir:
      outputdir = os.path.join(outputdir, self.options.data_type.name)
      if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    return os.path.join(outputdir, name)
  
  def write_spreadsheet(self, df, name):
    df.to_csv(self.create_output_name(name + ".csv"))

    df.to_pickle(self.create_output_name(name + ".pkl"))
    
    writer = pandas.ExcelWriter(self.create_output_name(name + ".xlsx"), engine="xlsxwriter")
    df.to_excel(writer)
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

  def run_prophet(self):
    import fbprophet
    results = {
      self.predict_column_name: [],
    }
    actions = self.get_possible_actions().tolist()
    for action in actions:
      df = self.get_action_data(action)
      
      df = df[[self.get_value_column_name()]]
      df.reset_index(inplace=True)
      df.rename(columns={"index": "ds", self.get_value_column_name(): "y"}, inplace=True)
      # https://facebook.github.io/prophet/docs/saturating_forecasts.html
      df["floor"] = 0

      prophet = fbprophet.Prophet()
      prophet.fit(df)
      future = prophet.make_future_dataframe(periods=self.options.predict, freq="Y")
      forecast = prophet.predict(future)

      prophet.plot(forecast)
      self.save_plot_image(action, "forecast")
      
      prophet.plot_components(forecast)
      self.save_plot_image(action, "components")

      forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
      results[self.predict_column_name].append(forecast.iloc[-1]["yhat"])

    return pandas.DataFrame(results, index=actions)

  def run_prepare_data(self):
    return False

  def run_test(self):
    return False

  def get_calculated_scale_function_values(self):
    return None

  def get_enum_names(self, e):
    return list(map(lambda x: x.name, list(e)))

  def run_get_action_data(self, action):
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

class DictTree(collections.defaultdict):
  def __init__(self, d={}, value=None, **kwargs):
    super().__init__(DictTree)
    self.update(d)
    self.update(kwargs)
    self.value = value
  
  def leaves_list(self):
    return self.leaves_list_process(self, [])
  
  def leaves_list_process(self, node, accumulator):
    for k, v in node.items():
      if isinstance(v, DictTree):
        accumulator = v.leaves_list_process(v, accumulator)
      elif isinstance(v, dict):
        accumulator = self.leaves_list_process(v, accumulator)
      else:
        accumulator.append(v)
    if self.value is not None and len(node) == 0:
      accumulator.append(self.value)
    return accumulator
  
  def flat_dict(self, leaves=False):
    result = {}
    self.flat_dict_process(self, result, leaves, None)
    return result
  
  def flat_dict_process(self, node, accumulator, leaves, key):
    for k, v in node.items():
      if isinstance(v, DictTree):
        v.flat_dict_process(v, accumulator, leaves, k)
      elif isinstance(v, dict):
        self.flat_dict_process(v, accumulator, leaves, k)
      else:
        accumulator[k] = v
    if key is not None and self.value is not None:
      if not leaves or (leaves and len(node) == 0):
        accumulator[key] = self.value
