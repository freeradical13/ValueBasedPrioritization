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
  ols_result = statsmodels.formula.api.ols(formula, data).fit()
  
  # http://www.statsmodels.org/dev/regression.html
  #
  # aic, bic, bse, centered_tss, compare_f_test, compare_lm_test,
  # compare_lr_test, condition_number, conf_int, conf_int_el, cov_HC0,
  # cov_HC1, cov_HC2, cov_HC3, cov_kwds, cov_params, cov_type,
  # df_model, df_resid, diagn, eigenvals, el_test, ess, f_pvalue,
  # f_test, fittedvalues, fvalue, get_influence, get_prediction,
  # get_robustcov_results, het_scale, initialize, k_constant, llf,
  # load, model, mse_model, mse_resid, mse_total, nobs,
  # normalized_cov_params, outlier_test, params, predict, pvalues,
  # remove_data, resid, resid_pearson, rsquared, rsquared_adj, save,
  # scale, ssr, summary, summary2, t_test, t_test_pairwise, tvalues,
  # uncentered_tss, use_t, wald_test, wald_test_terms, wresid
  
  return ols_result

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
