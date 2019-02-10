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

def linear_regression(df, x, y, degree=1):
  data = {"x": df[x].values, "y": df[y].values}
  formula = "y ~ " + " + ".join("I(x**{})".format(i) for i in range(1, degree+1))
  print("Formula: {}".format(formula))
  return statsmodels.formula.api.ols(formula, data).fit()
