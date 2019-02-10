# Run linear regressions on Underlying Cause of Death statistics
# https://wonder.cdc.gov/ucd-icd10.html
# Group Results By “Year” And By “ICD Sub-Chapter”; Check
# “Export Results”; Uncheck “Show Totals”

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

import vbp.ucod.us

def run_regressions(args):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("cause", help="ICD Sub-Chapter")
  parser.add_argument("-f", "--file", help="path to file", default="data/Underlying Cause of Death, 1999-2017_ICD10_Sub-Chapters.txt")
  parser.add_argument("-m", "--min-degrees", help="minimum polynomial degree", type=int, default=1)
  parser.add_argument("-x", "--max-degrees", help="maximum polynomial degree", type=int, default=4)
  parser.add_argument("-p", "--predict", help="future prediction (years)", type=int, default=5)
  options = parser.parse_args(args)
  
  ds = vbp.ucod.us.UnderyingCausesOfDeathUnitedStates()
  ds.load(file=options.file)
  for i in range(options.min_degrees, options.max_degrees + 1):
    ds.create_plot(options.cause, options.predict, degree=i)

if __name__ == "__main__":
  run_regressions(sys.argv[1:])
