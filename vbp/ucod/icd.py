import numpy

# https://en.wikipedia.org/wiki/International_Statistical_Classification_of_Diseases_and_Related_Health_Problems
class ICD:
  @staticmethod
  def toint(x, addone=False):
    x = x.replace("-", "")
    if len(x) == 4:
      x = x[0:3]
    x = ICD.normalize(x)
    result = int(x)
    if addone:
      result += 1
    return result
  
  @staticmethod
  def tofloat(x):
    x = x.replace("-", "")
    f = 0
    if len(x) == 4:
      f = float(x[3]) / 10.0
      x = x[0:3]
    x = ICD.normalize(x)
    return float(x) + f

  @staticmethod
  def blockint(c):
    return (ord(c) - 64) * 100

  @staticmethod
  def normalize(x):
    if x[0].isalpha():
      x = str(ICD.blockint(x[0])) + x[1:]
    return x
  
  @staticmethod
  def convert_age_minutes(x):
    if len(x) < 3:
      return int(int(x) * 365.25 * 24 * 60) # years
    else:
      ageType = x[0]
      if ageType == '0' or ageType == '1': # years
        return int(int(x[1:]) * 365.25 * 24 * 60)
      elif ageType == '2': # months
        return int(int(x[1:]) * ((365.25/12) * 24 * 60))
      elif ageType == '3': # weeks
        return int(int(x[1:]) * ((365.25/52) * 7 * 24 * 60))
      elif ageType == '4': # days
        return int(x[1:]) * 24 * 60
      elif ageType == '5': # hours
        return int(x[1:]) * 60
      elif ageType == '6': # minutes
        return int(x[1:])
      elif ageType == '9': # Age unknown
        return numpy.NaN
      else:
        raise ValueError("Unexpected age {}".format(x))
