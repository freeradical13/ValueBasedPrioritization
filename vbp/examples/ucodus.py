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

if __name__ == "__main__":
  ds = vbp.ucod.us.UnderlyingCausesOfDeathUnitedStates()
  ds.predict(sys.argv[1:])
