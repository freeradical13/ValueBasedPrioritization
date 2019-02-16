import os
import abc
import sys
import math
import numpy
import pandas
import random
import shutil
import pathlib
import argparse
import datetime
import matplotlib
import matplotlib.offsetbox
import statsmodels.tools
import statsmodels.formula.api

VERSION = "0.1.0"

def linear_regression_formula(degree=1):
  return "y ~ " + " + ".join("I(x**{})".format(i) for i in range(1, degree+1))

def linear_regression(df, x, y, formula):
  data = {"x": df[x].values, "y": df[y].values}

  # https://www.statsmodels.org/dev/regression.html
  # https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
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
  # (((element - min(nparray)) * (max - min)) / (max(nparray) - min(nparray))) + min
  min_nparray = nparray.min()
  max_nparray = nparray.max()
  minmax_nparray_diff = max_nparray - min_nparray
  minmaxdiff = max - min
  # (((element - min_nparray) * minmaxdiff) / minmax_nparray_diff) + min
  #return numpy.vectorize(lambda x: ((x - min_nparray) * minmaxdiff) / minmax_nparray_diff)(nparray) + min
  # =>
  # ((element * minmaxdiff) - (min_nparray * minmaxdiff)) / minmax_nparray_diff) + min
  min_nparray_minmaxdiff = min_nparray * minmaxdiff
  return numpy.vectorize(lambda x: (((x * minmaxdiff) - min_nparray_minmaxdiff) / minmax_nparray_diff) + min)(nparray)

def normalize0to1(nparray):
  return normalize(nparray, 0, 1)

class DetailedErrorArgumentParser(argparse.ArgumentParser):
  def error(self, message):
    sys.stderr.write("error: {}\n\n".format(message))
    self.print_help()
    sys.exit(1)

def create_parser():
  return DetailedErrorArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class DataSource(object, metaclass=abc.ABCMeta):
  ##############
  # Attributes #
  ##############
  obfuscated_column_name = "Action"
  obfuscated_action_names = {}
  cached_obfuscation_range = None

  #######################
  # Main Public Methods #
  #######################
  def load(self, args):
    self.prepare(args)
    return self
  
  def predict(self):
    return self.run_predict()

  def get_possible_actions(self):
    return self.run_get_possible_actions()
  
  def get_action_data(self, action):
    return self.run_get_action_data(action)

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
      parser.add_argument("-o", "--output-dir", help="output directory", default="output")
      parser.add_argument("--clean", action="store_true", help="first clean any existing generated data such as images", default=False)
      parser.add_argument("--do-not-obfuscate", action="store_true", help="do not obfuscate action names", default=False)
      self.options = parser.parse_args(args)
      if self.options.clean:
        self.clean_files()
      if not os.path.exists(self.options.output_dir):
        os.makedirs(self.options.output_dir)

  def ensure_loaded(self):
    if not hasattr(self, "data"):
      self.data = self.run_load()
      self.post_process()
  
  def post_process(self):
    self.data[self.obfuscated_column_name] = self.data[self.get_action_column_name()].apply(lambda x: self.get_obfuscated_name(x))

  def run_get_possible_actions(self):
    return self.data[self.get_action_column_name()].unique()

  def run_get_action_data(self, action):
    return self.data[self.data[self.get_action_column_name()] == action]
    
  def get_obfuscated_name(self, str):
    if self.options.do_not_obfuscate:
      return str
    result = self.obfuscated_action_names.get(str)
    if result is None:
      if self.cached_obfuscation_range is None:
        self.cached_obfuscation_range = len(self.get_possible_actions())
      result = "{}{}".format(self.obfuscated_column_name, random.randint(1, self.cached_obfuscation_range))
      self.obfuscated_action_names[str] = result
    return result

  def clean_files(self):
    shutil.rmtree(self.options.output_dir, ignore_errors=True)

  def create_output_name(self, name):
    return os.path.join(self.options.output_dir, name)

def print_full_columns(df):
  with pandas.option_context("display.max_columns", None):
    with pandas.option_context("display.max_colwidth", sys.maxsize):
      print(df)

