import abc
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

VERSION = "0.1.0"

def linear_regression(df, x, y, degree=1):
  data = {"x": df[x].values, "y": df[y].values}
  formula = "y ~ " + " + ".join("I(x**{})".format(i) for i in range(1, degree+1))
  print("Formula: {}".format(formula))
  return statsmodels.formula.api.ols(formula, data).fit()

class DetailedErrorArgumentParser(argparse.ArgumentParser):
  def error(self, message):
    sys.stderr.write("error: {}\n\n".format(message))
    self.print_help()
    sys.exit(1)

def create_parser():
  return DetailedErrorArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class DataSource(object, metaclass=abc.ABCMeta):
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

  #################
  # Other Methods #
  #################
  @abc.abstractmethod
  def run_load(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def run_predict(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def run_get_possible_actions(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def run_get_action_data(self, action):
    raise NotImplementedError()

  @abc.abstractmethod
  def initialize_parser(self, parser):
    raise NotImplementedError()

  def prepare(self, args):
    self.ensure_options(args)
    self.ensure_loaded()

  def ensure_options(self, args):
    if not hasattr(self, "options"):
      parser = create_parser()
      self.initialize_parser(parser)
      self.options = parser.parse_args(args)

  def ensure_loaded(self):
    if not hasattr(self, "data"):
      self.data = self.run_load()
