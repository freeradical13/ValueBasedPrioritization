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
import vbp.ucod.icd
import urllib.request
import matplotlib.pyplot
import statsmodels.tools
import statsmodels.tsa.api
import matplotlib.offsetbox
import statsmodels.stats.api
import statsmodels.formula.api
import statsmodels.stats.diagnostic
import statsmodels.tools.eval_measures

from vbp import DictTree
from vbp.ucod.icd import ICD

class DataType(vbp.DataSourceDataType):
  
  # https://www.cdc.gov/nchs/data/dvs/Multiple_Cause_Record_Layout_2016.pdf (Page 19)
  US_ICD_113_SELECTED_CAUSES_ALL = enum.auto()

  # Same as above but only the leaves of the tree. For some reason, this is not exactly 113, but 118.
  US_ICD_113_SELECTED_CAUSES_LEAVES = enum.auto()

  # Same as above but only the roots of the tree.
  US_ICD_113_SELECTED_CAUSES_ROOTS = enum.auto()

  # Group Results By "Year" And By "ICD Chapter"; Check "Export Results"; Uncheck "Show Totals"
  # https://wonder.cdc.gov/ucd-icd10.html
  US_ICD10_CHAPTERS = enum.auto()
  
  # Group Results By "Year" And By "ICD Sub-Chapter"; Check "Export Results"; Uncheck "Show Totals"
  # https://wonder.cdc.gov/ucd-icd10.html
  US_ICD10_SUB_CHAPTERS = enum.auto()

  # Group Results By "Year" And By "Cause of death"; Check "Export Results"; Uncheck "Show Totals"
  # https://wonder.cdc.gov/ucd-icd10.html
  US_ICD10_MINIMALLY_GROUPED = enum.auto()
  
  # Built from https://www.cdc.gov/nchs/data/dvs/lead1900_98.pdf and Mortality data >= 1959 and comparability ratios
  # from https://www.cdc.gov/nchs/data/dvs/comp2.pdf
  US_ICD_LONGTERM_COMPARABLE_LEADING = enum.auto()

class UCODUnitedStates(vbp.ucod.icd.ICDDataSource):

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
  
  # https://www.cdc.gov/nchs/data/dvs/Multiple_Cause_Record_Layout_2016.pdf (Page 19)
  icd10_ucod113 = DictTree({
    1: "Salmonella infections (A01-A02)",
    2: "Shigellosis and amebiasis (A03,A06)",
    3: "Certain other intestinal infections (A04,A07-A09)",
    4: DictTree(value="Tuberculosis (A16-A19)", d={
      5: "Respiratory tuberculosis (A16)",
      6: "Other tuberculosis (A17-A19)",
    }),
    7: "Whooping cough (A37)",
    8: "Scarlet fever and erysipelas (A38,A46)",
    9: "Meningococcal infection (A39)",
    10: "Septicemia (A40-A41)",
    11: "Syphilis (A50-A53)",
    12: "Acute poliomyelitis (A80)",
    13: "Arthropod-borne viral encephalitis (A83-A84,A85.2)",
    14: "Measles (B05)",
    15: "Viral hepatitis (B15-B19)",
    16: "Human immunodeficiency virus (HIV) disease (B20-B24)",
    17: "Malaria (B50-B54)",
    18: "Other and unspecified infectious and parasitic diseases and their sequelae (A00,A05,A20-A36,A42-A44,A48-A49,A54-A79,A81-A82,A85.0-A85.1,A85.8,A86-B04,B06-B09,B25-B49,B55-B99)",
    19: DictTree(value="Malignant neoplasms (C00-C97)", d={
      20: "Malignant neoplasms of lip, oral cavity and pharynx (C00-C14)",
      21: "Malignant neoplasm of esophagus (C15)",
      22: "Malignant neoplasm of stomach (C16)",
      23: "Malignant neoplasms of colon, rectum and anus (C18-C21)",
      24: "Malignant neoplasms of liver and intrahepatic bile ducts (C22)",
      25: "Malignant neoplasm of pancreas (C25)",
      26: "Malignant neoplasm of larynx (C32)",
      27: "Malignant neoplasms of trachea, bronchus and lung (C33-C34)",
      28: "Malignant melanoma of skin (C43)",
      29: "Malignant neoplasm of breast (C50)",
      30: "Malignant neoplasm of cervix uteri (C53)",
      31: "Malignant neoplasms of corpus uteri and uterus, part unspecified (C54-C55)",
      32: "Malignant neoplasm of ovary (C56)",
      33: "Malignant neoplasm of prostate (C61)",
      34: "Malignant neoplasms of kidney and renal pelvis (C64-C65)",
      35: "Malignant neoplasm of bladder (C67)",
      36: "Malignant neoplasms of meninges, brain and other parts of central nervous system (C70-C72)",
      37: DictTree(value="Malignant neoplasms of lymphoid, hematopoietic and related tissue (C81-C96)", d={
        38: "Hodgkin's disease (C81)",
        39: "Non-Hodgkin's lymphoma (C82-C85)",
        40: "Leukemia (C91-C95)",
        41: "Multiple myeloma and immunoproliferative neoplasms (C88,C90)",
        42: "Other and unspecified malignant neoplasms of lymphoid, hematopoietic and related tissue (C96)",
      }),
      43: "All other and unspecified malignant neoplasms (C17,C23-C24,C26-C31,C37-C41,C44-C49,C51-C52,C57-C60,C62-C63,C66,C68-C69,C73-C80,C97)",
    }),
    44: "In situ neoplasms, benign neoplasms and neoplasms of uncertain or unknown behavior (D00-D48)",
    45: "Anemias (D50-D64)",
    46: "Diabetes mellitus (E10-E14)",
    47: DictTree(value="Nutritional deficiencies (E40-E64)", d={
      48: "Malnutrition (E40-E46)",
      49: "Other nutritional deficiencies (E50-E64)",
    }),
    50: "Meningitis (G00,G03)",
    51: "Parkinson's disease (G20-G21)",
    52: "Alzheimer's disease (G30)",
    53: DictTree(value="Major cardiovascular diseases (I00-I78)", d={
      54: DictTree(value="Diseases of heart (I00-I09,I11,I13,I20-I51)", d={
        55: "Acute rheumatic fever and chronic rheumatic heart diseases (I00-I09)",
        56: "Hypertensive heart disease (I11)",
        57: "Hypertensive heart and renal disease (I13)",
        58: "Ischemic heart diseases (I20-I25)",
        59: "Acute myocardial infarction (I21-I22)",
        60: "Other acute ischemic heart diseases (I24)",
        61: DictTree(value="Other forms of chronic ischemic heart disease (I20,I25)", d={
          62: "Atherosclerotic cardiovascular disease, so described (I25.0)",
          63: "All other forms of chronic ischemic heart disease (I20,I25.1-I25.9)",
        }),
        64: DictTree(value="Other heart diseases (I26-I51)", d={
          65: "Acute and subacute endocarditis (I33)",
          66: "Diseases of pericardium and acute myocarditis (I30-I31,I40)",
          67: "Heart failure (I50)",
          68: "All other forms of heart disease (I26-I28,I34-I38,I42-I49,I51)",
        }),
      }),
      69: "Essential (primary) hypertension and hypertensive renal disease (I10,I12,I15)",
      70: "Cerebrovascular diseases (I60-I69)",
      71: "Atherosclerosis (I70)",
      72: DictTree(value="Other diseases of circulatory system (I71-I78)", d={
        73: "Aortic aneurysm and dissection (I71)",
        74: "Other diseases of arteries, arterioles and capillaries (I72-I78)",
      }),
    }),
    75: "Other disorders of circulatory system (I80-I99)",
    76: DictTree(value="Influenza and pneumonia (J09-J18)", d={
      77: "Influenza (J09-J11)",
      78: "Pneumonia (J12-J18)",
    }),
    79: DictTree(value="Other acute lower respiratory infections (J20-J22,U04)", d={
      80: "Acute bronchitis and bronchiolitis (J20-J21)",
      81: "Other and unspecified acute lower respiratory infection (J22,U04)",
    }),
    82: DictTree(value="Chronic lower respiratory diseases (J40-J47)", d={
      83: "Bronchitis, chronic and unspecified (J40-J42)",
      84: "Emphysema (J43)",
      85: "Asthma (J45-J46)",
      86: "Other chronic lower respiratory diseases (J44,J47)",
    }),
    87: "Pneumoconioses and chemical effects (J60-J66,J68)",
    88: "Pneumonitis due to solids and liquids (J69)",
    89: "Other diseases of respiratory system (J00-J06,J30-J39,J67,J70-J98)",
    90: "Peptic ulcer (K25-K28)",
    91: "Diseases of appendix (K35-K38)",
    92: "Hernia (K40-K46)",
    93: DictTree(value="Chronic liver disease and cirrhosis (K70,K73-K74)", d={
      94: "Alcoholic liver disease (K70)",
      95: "Other chronic liver disease and cirrhosis (K73-K74)",
    }),
    96: "Cholelithiasis and other disorders of gallbladder (K80-K82)",
    97: DictTree(value="Nephritis, nephrotic syndrome and nephrosis (N00-N07,N17-N19,N25-N27)", d={
      98: "Acute and rapidly progressive nephritic and nephrotic syndrome (N00-N01,N04)",
      99: "Chronic glomerulonephritis, nephritis and nephropathy not specified as acute or chronic, and renal sclerosis unspecified (N02-N03,N05-N07,N26)",
      100: "Renal failure (N17-N19)",
      101: "Other disorders of kidney (N25,N27)",
      102: "Infections of kidney (N10-N12,N13.6,N15.1)",
    }),
    103: "Hyperplasia of prostate (N40)",
    104: "Inflammatory diseases of female pelvic organs (N70-N76)",
    105: DictTree(value="Pregnancy, childbirth and the puerperium (O00-O99)", d={
      106: "Pregnancy with abortive outcome (O00-O07)",
      107: "Other complications of pregnancy, childbirth and the puerperium (O10-O99)",
    }),
    108: "Certain conditions originating in the perinatal period (P00-P96)",
    109: "Congenital malformations, deformations and chromosomal abnormalities (Q00-Q99)",
    110: "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)",
    111: "All other diseases (Residual) (NaN)",
    112: DictTree(value="Accidents (unintentional injuries) (V01-X59,Y85-Y86)", d={
      113: "Transport accidents (V01-V99,Y85)",
      114: "Motor vehicle accidents (V02-V04,V09.0,V09.2,V12-V14,V19.0-V19.2,V19.4-V19.6,V20-V79,V80.3-V80.5,V81.0-V81.1,V82.0-V82.1,V83-V86, V87.0-V87.8,V88.0-V88.8,V89.0,V89.2)",
      115: "Other land transport accidents (V01,V05-V06,V09.1,V09.3-V09.9, V10-V11,V15-V18,V19.3,V19.8-V19.9,V80.0-V80.2,V80.6-V80.9,V81.2-V81.9,V82.2-V82.9,V87.9,V88.9,V89.1,V89.3,V89.9)",
      116: "Water, air and space, and other and unspecified transport accidents and their sequelae (V90-V99,Y85)",
      117: "Nontransport accidents (W00-X59,Y86)",
      118: "Falls (W00-W19)",
      119: "Accidental discharge of firearms (W32-W34)",
      120: "Accidental drowning and submersion (W65-W74)",
      121: "Accidental exposure to smoke, fire and flames (X00-X09)",
      122: "Accidental poisoning and exposure to noxious substances (X40-X49)",
      123: "Other and unspecified nontransport accidents and their sequelae (W20-W31,W35-W64,W75-W99,X10-X39,X50-X59,Y86)",
    }),
    124: DictTree(value="Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)", d={
      125: "Intentional self-harm (suicide) by discharge of firearms (X72-X74)",
      126: "Intentional self-harm (suicide) by other and unspecified means and their sequelae (*U03,X60-X71,X75-X84,Y87.0)",
    }),
    127: DictTree(value="Assault (homicide) (*U01-*U02,X85-Y09,Y87.1)", d={
      128: "Assault (homicide) by discharge of firearms (*U01.4,X93-X95)",
      129: "Assault (homicide) by other and unspecified means and their sequelae (*U01.0-*U01.3,*U01.5-*U01.9,*U02,X85-X92,X96-Y09,Y87.1)",
    }),
    130: "Legal intervention (Y35,Y89.0)",
    131: "Events of undetermined intent (Y10-Y34,Y87.2,Y89.9)",
    132: "Discharge of firearms, undetermined intent (Y22-Y24)",
    133: "Other and unspecified events of undetermined intent and their sequelae (Y10-Y21,Y25-Y34,Y87.2,Y89.9)",
    134: "Operations of war and their sequelae (Y36,Y89.1)",
    135: "Complications of medical and surgical care (Y40-Y84,Y88)",
  })

  # https://www.nber.org/mortality/9_10/ICD9_ICD10_comparability_file_documentation.pdf
  # https://www.cdc.gov/nchs/nvss/mortality/comparability_icd.htm
  icd9_ucod113 = DictTree({
    1: "Salmonella infections (002-003)",
    2: "Shigellosis and amebiasis (004,006)",
    3: "Certain other intestinal infections (007-009)",
    4: DictTree(value="Tuberculosis (010-018)", d={
      5: "Respiratory tuberculosis (010-012)",
      6: "Other tuberculosis (013-018)",
    }),
    7: "Whooping cough (033)",
    8: "Scarlet fever and erysipelas (034.1-035)",
    9: "Meningococcal infection (036)",
    10: "Septicemia (038)",
    11: "Syphilis (090-097)",
    12: "Acute poliomyelitis (045)",
    13: "Arthropod-borne viral encephalitis (062-064)",
    14: "Measles (055)",
    15: "Viral hepatitis (070)",
    16: "Human immunodeficiency virus (HIV) disease (042-044)",
    17: "Malaria (084)",
    18: "Other and unspecified infectious and parasitic diseases and their sequelae (001,005,020-032,037,039-041,046-054,056-061,065-066,071-083,085-088,098-134,136,139,771.3)",
    19: DictTree(value="Malignant neoplasms (140-208)", d={
      20: "Malignant neoplasms of lip, oral cavity and pharynx (140-149)",
      21: "Malignant neoplasm of esophagus (150)",
      22: "Malignant neoplasm of stomach (151)",
      23: "Malignant neoplasms of colon, rectum and anus (153-154)",
      24: "Malignant neoplasms of liver and intrahepatic bile ducts (155)",
      25: "Malignant neoplasm of pancreas (157)",
      26: "Malignant neoplasm of larynx (161)",
      27: "Malignant neoplasms of trachea, bronchus and lung (162)",
      28: "Malignant melanoma of skin (172)",
      29: "Malignant neoplasm of breast (174-175)",
      30: "Malignant neoplasm of cervix uteri (180)",
      31: "Malignant neoplasms of corpus uteri and uterus, part unspecified (179,182)",
      32: "Malignant neoplasm of ovary (183.0)",
      33: "Malignant neoplasm of prostate (185)",
      34: "Malignant neoplasms of kidney and renal pelvis (189.0,189.1)",
      35: "Malignant neoplasm of bladder (188)",
      36: "Malignant neoplasms of meninges, brain and other parts of central nervous system (191-192)",
      37: DictTree(value="Malignant neoplasms of lymphoid, hematopoietic and related tissue (200-208)", d={
        38: "Hodgkin's disease (201)",
        39: "Non-Hodgkin's lymphoma (200,202)",
        40: "Leukemia (204-208)",
        41: "Multiple myeloma and immunoproliferative neoplasms (203)",
        42: "Other and unspecified malignant neoplasms of lymphoid, hematopoietic and related tissue (NaN)",
      }),
      43: "All other and unspecified malignant neoplasms (152,156,158-160,163-171,173,181,183.2-184,186-187,189.2-190,193-199)",
    }),
    44: "In situ neoplasms, benign neoplasms and neoplasms of uncertain or unknown behavior (210-239)",
    45: "Anemias (280-285)",
    46: "Diabetes mellitus (250)",
    47: DictTree(value="Nutritional deficiencies (260-269)", d={
      48: "Malnutrition (260-263)",
      49: "Other nutritional deficiencies (264-269)",
    }),
    50: "Meningitis (320-322)",
    51: "Parkinson's disease (332)",
    52: "Alzheimer's disease (331.0)",
    53: DictTree(value="Major cardiovascular diseases (390-434,436-448)", d={
      54: DictTree(value="Diseases of heart (390-398,402,404,410-429)", d={
        55: "Acute rheumatic fever and chronic rheumatic heart diseases (390-398)",
        56: "Hypertensive heart disease (402)",
        57: "Hypertensive heart and renal disease (404)",
        58: "Ischemic heart diseases (410-414,429.2)",
        59: "Acute myocardial infarction (410)",
        60: "Other acute ischemic heart diseases (411)",
        61: DictTree(value="Other forms of chronic ischemic heart disease (412-414,429.2)", d={
          62: "Atherosclerotic cardiovascular disease, so described (429.2)",
          63: "All other forms of chronic ischemic heart disease (412-414)",
        }),
        64: DictTree(value="Other heart diseases (415-429.1,429.3-429.9)", d={
          65: "Acute and subacute endocarditis (421)",
          66: "Diseases of pericardium and acute myocarditis (420,422-423)",
          67: "Heart failure (428)",
          68: "All other forms of heart disease (415-417,424-427,429.0-429.1,429.3-429.9)",
        }),
      }),
      69: "Essential (primary) hypertension and hypertensive renal disease (401,403)",
      70: "Cerebrovascular diseases (430-434,436-438)",
      71: "Atherosclerosis (440)",
      72: DictTree(value="Other diseases of circulatory system (441-448)", d={
        73: "Aortic aneurysm and dissection (441)",
        74: "Other diseases of arteries, arterioles and capillaries (442-448)",
      }),
    }),
    75: "Other disorders of circulatory system (451-459)",
    76: DictTree(value="Influenza and pneumonia (480-487)", d={
      77: "Influenza (487)",
      78: "Pneumonia (480-486)",
    }),
    79: DictTree(value="Other acute lower respiratory infections (466)", d={
      80: "Acute bronchitis and bronchiolitis (466)",
      81: "Other and unspecified acute lower respiratory infection (NaN)",
    }),
    82: DictTree(value="Chronic lower respiratory diseases (490-494,496)", d={
      83: "Bronchitis, chronic and unspecified (490-491)",
      84: "Emphysema (492)",
      85: "Asthma (493)",
      86: "Other chronic lower respiratory diseases (494,496)",
    }),
    87: "Pneumoconioses and chemical effects (500-506)",
    88: "Pneumonitis due to solids and liquids (507)",
    89: "Other diseases of respiratory system (034.0,460-465,470-478,495,508-519)",
    90: "Peptic ulcer (531-534)",
    91: "Diseases of appendix (540-543)",
    92: "Hernia (550-553)",
    93: DictTree(value="Chronic liver disease and cirrhosis (571)", d={
      94: "Alcoholic liver disease (571.0-571.3)",
      95: "Other chronic liver disease and cirrhosis (571.4-571.9)",
    }),
    96: "Cholelithiasis and other disorders of gallbladder (574-575)",
    97: DictTree(value="Nephritis, nephrotic syndrome and nephrosis (580-589)", d={
      98: "Acute and rapidly progressive nephritic and nephrotic syndrome (580-581)",
      99: "Chronic glomerulonephritis, nephritis and nephropathy not specified as acute or chronic, and renal sclerosis unspecified (582-583,587)",
      100: "Renal failure (584-586)",
      101: "Other disorders of kidney (588-589)",
      102: "Infections of kidney (590)",
    }),
    103: "Hyperplasia of prostate (600)",
    104: "Inflammatory diseases of female pelvic organs (614-616)",
    105: DictTree(value="Pregnancy, childbirth and the puerperium (630-676)", d={
      106: "Pregnancy with abortive outcome (630-639)",
      107: "Other complications of pregnancy, childbirth and the puerperium (640-676)",
    }),
    108: "Certain conditions originating in the perinatal period (760-771.2,771.4-779)",
    109: "Congenital malformations, deformations and chromosomal abnormalities (740-759)",
    110: "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (780-799)",
    111: "All other diseases (Residual) (NaN)",
    112: DictTree(value="Accidents (unintentional injuries) (800-869,880-929)", d={
      113: "Transport accidents (800-848,929.0,929.1)",
      114: "Motor vehicle accidents (810-825)",
      115: "Other land transport accidents (800-807,826-829)",
      116: "Water, air and space, and other and unspecified transport accidents and their sequelae (830-848,929.0,929.1)",
      117: "Nontransport accidents (850-869,880-928,929.2-929.9)",
      118: "Falls (880-888)",
      119: "Accidental discharge of firearms (922)",
      120: "Accidental drowning and submersion (910)",
      121: "Accidental exposure to smoke, fire and flames (890-899)",
      122: "Accidental poisoning and exposure to noxious substances (850-869,924.1)",
      123: "Other and unspecified nontransport accidents and their sequelae (900-909,911-921,923-924.0,924.8-928,929.2-929.9)",
    }),
    124: DictTree(value="Intentional self-harm (suicide) (950-959)", d={
      125: "Intentional self-harm (suicide) by discharge of firearms (955.0-955.4)",
      126: "Intentional self-harm (suicide) by other and unspecified means and their sequelae (950-954,955.5-959)",
    }),
    127: DictTree(value="Assault (homicide) (960-969)", d={
      128: "Assault (homicide) by discharge of firearms (965.0-965.4)",
      129: "Assault (homicide) by other and unspecified means and their sequelae (960-964,965.5-969)",
    }),
    130: "Legal intervention (970-978)",
    131: "Events of undetermined intent (980-989)",
    132: "Discharge of firearms, undetermined intent (985.0-985.4)",
    133: "Other and unspecified events of undetermined intent and their sequelae (980-984,985.5-989)",
    134: "Operations of war and their sequelae (990-999)",
    135: "Complications of medical and surgical care (870-879,930-949)",
  })
  
  def initialize_parser(self, parser):
    super().initialize_parser(parser)
    parser.add_argument("--average-ages", help="Compute average ages column with the specified column name", default="AverageAge")
    parser.add_argument("--average-age-range", help="Range over which to calculate the average age", type=int, default=5)
    parser.add_argument("--comparable-ratios", help="Process comparable ratios for raw mortality matrix for prepare_data", action="store_true", default=False)
    parser.add_argument("--data-comparable-ratios-input-file", help="Comparable ratios file", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/comparable_ucod_estimates.xlsx"))
    parser.add_argument("--data-comparability-ratio-tables", help="Comparable ratios file", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/Comparability_Ratio_tables.xls"))
    parser.add_argument("--data-us-icd10-sub-chapters", help="Path to file for US_ICD10_SUB_CHAPTERS", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/Underlying Cause of Death, 1999-2017_US_ICD10_SUB_CHAPTERS.txt"))
    parser.add_argument("--data-us-icd10-chapters", help="Path to file for US_ICD10_CHAPTERS", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/Underlying Cause of Death, 1999-2017_US_ICD10_CHAPTERS.txt"))
    parser.add_argument("--data-us-icd10-minimally-grouped", help="Path to file for US_ICD10_MINIMALLY_GROUPED", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/Underlying Cause of Death, 1999-2017_US_ICD10_MINIMALLY_GROUPED.txt"))
    parser.add_argument("--data-us-icd10-113-selected-causes", help="Path to file for US_ICD10_113_SELECTED_CAUSES", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/Underlying Cause of Death, 1999-2017_US_ICD10_113_SELECTED_CAUSES.txt"))
    parser.add_argument("--data-us-icd-longterm-comparable-leading", help="Path to file for US_ICD_LONGTERM_COMPARABLE_LEADING", default=os.path.join(self.get_data_dir(), "data/ucod/united_states/comparable_ucod_estimates_ratios_applied.xlsx"))
    parser.add_argument("--download", help="If no files in --raw-files-directory, download and extract", action="store_true", default=True)
    parser.add_argument("--raw-files-directory", help="directory with raw files", default=os.path.join(self.default_cache_dir, "united_states"))
    parser.add_argument("--test", help="Test")

  @staticmethod
  def get_data_types_enum():
    return DataType

  @staticmethod
  def get_data_types_enum_default():
    return DataType.US_ICD_LONGTERM_COMPARABLE_LEADING

  def run_load(self):
    if self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
         
      df = self.get_raw_data_selected_causes()
      
      if self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
        
        if self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES:
          target = self.icd9_ucod113.recursive_list(True) + self.icd10_ucod113.recursive_list(True)
        else:
          target = self.icd9_ucod113.roots_list() + self.icd10_ucod113.roots_list()
          
        keep_queries = list(map(self.icd_query, map(self.extract_codes, target)))
        df["CodesQuery"] = df[self.get_code_column_name()].apply(self.icd_query)
        
        # Drop any non-leaves
        df.drop(df[~df["CodesQuery"].isin(keep_queries)].index, inplace=True)
        
        # The column ends up being a string column, so we can't just do dropna()
        df.drop(df[df[self.get_code_column_name()] == "NaN"].index, inplace=True)
        
    elif self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS or \
         self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED or \
         self.options.data_type == DataType.US_ICD10_CHAPTERS:

      df = pandas.read_csv(
             self.get_data_file(),
             sep="\t",
             usecols=["Year", "Deaths", "Population"] + self.get_read_columns(),
             na_values=["Unreliable"],
             parse_dates=[0],
             encoding="ISO-8859-1",
           ).dropna(how="all")

      if self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED:
        # Lookup by action_column_name but these aren't unique in this data set,
        # so append the codes.
        df[self.get_action_column_name()] = df.apply(lambda row: "{} ({})".format(row[self.get_action_column_name()], row[self.get_code_column_name()]), axis="columns")

    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:

      df = pandas.read_excel(
        self.get_data_file(),
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
    df[self.get_value_column_name()] = (df.Deaths / df.Population) * self.crude_rate_amount()
    self.write_spreadsheet(df, self.prefix_all("data"))
    return df

  def load_us_raw_data_with_comparability_ratios(self):
    self.check_raw_files_directory()
    df = self.get_raw_mortality_counts(
      self.raw_icd_basis,
      None,
      None,
      self.raw_icd9_counts,
      self.raw_icd10_counts,
      min_year=1979, # The start of ICD9 which is all we have easy access to compatability ratios for
      calculate_total=False,
      add_icd_revision=False,
      comparability_ratios=self.get_icd9_10_comparability_ratios(),
    )
    df.index.name = "Year"
    df = df.reset_index().melt(id_vars=["Year"], value_vars=df.columns.values).sort_values("Year")
    df.columns = ["Year", self.get_action_column_name(), "Deaths"]
    df[self.get_code_column_name()] = df.apply(lambda row: self.extract_codes((UCODUnitedStates.icd10_ucod113 if row["Year"] >= 1999 else UCODUnitedStates.icd9_ucod113).find_value(lambda v: self.extract_name(v) == row[self.get_action_column_name()])), axis="columns")
    df["Population"] = df.Year.apply(lambda y: self.mortality_uspopulation.loc[y]["Population"])
    df["Year"] = df["Year"].apply(lambda year: pandas.datetime.strptime(str(year), "%Y"))
    return df

  def get_raw_data_selected_causes(self):
    return self.load_with_cache("us_raw_data_with_comparability_ratios", self.load_us_raw_data_with_comparability_ratios)
    
  def get_action_column_name(self):
    if self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS:
      return "ICD Sub-Chapter"
    elif self.options.data_type == DataType.US_ICD10_CHAPTERS:
      return "ICD Chapter"
    elif self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED:
      return "Cause of death"
    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:
      return "Cause of death"
    elif self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
      return "ICD-10 113 Cause List"
    else:
      raise NotImplementedError()
  
  def get_value_column_name(self):
    return "Crude Rate" + super().get_value_column_name()
  
  def get_code_column_name(self):
    if self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS:
      return "ICD Sub-Chapter Code"
    elif self.options.data_type == DataType.US_ICD10_CHAPTERS:
      return "ICD Chapter Code"
    elif self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED:
      return "Cause of death Code"
    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:
      return "Cause of death Code"
    elif self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
      return "Cause of death Codes"
    else:
      raise NotImplementedError()
  
  def get_data_file(self):
    if self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS:
      return self.options.data_us_icd10_sub_chapters
    elif self.options.data_type == DataType.US_ICD10_CHAPTERS:
      return self.options.data_us_icd10_chapters
    elif self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED:
      return self.options.data_us_icd10_minimally_grouped
    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:
      return self.options.data_us_icd_longterm_comparable_leading
    elif self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
         self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
      return self.options.data_us_icd10_113_selected_causes
    else:
      raise NotImplementedError()
  
  def get_read_columns(self):
    if self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS:
      return [self.get_action_column_name()]
    else:
      return [self.get_action_column_name(), self.get_code_column_name()]
    
  def get_action_title_prefix(self):
    return "Deaths from "
  
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
    self.options.data_type = DataType.US_ICD_LONGTERM_COMPARABLE_LEADING
    self.check_raw_files_directory()
    if self.options.comparable_ratios:
      self.create_comparable()
    else:
      self.process_raw_mortality_data()
    
  def get_mortality_files(self):
    return sorted(glob.glob(os.path.join(self.options.raw_files_directory, "*.csv")))
  
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
    df["icdint"] = df["ucod"].apply(ICD.toint)
    df["icdfloat"] = df["ucod"].apply(ICD.tofloat)
    
    scale = 1
    if file_year == 1972:
      # "The 1972 file is a 50% sample" (https://www.nber.org/mortality/errata.txt)
      # Same in raw data: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/mort1972us.zip
      scale = 2
    
    return df, scale
  
  def get_longterm_comparable_yearly_basis(self):
    return {
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
  
  def longterm_comparable_icd7(self, df, count_years, scale, comparability_ratios):
    count_years["Tuberculosis"] = len(df.query(self.icd_query("001-019")))*scale
    count_years["Diarrhea, enteritis, and colitis"] = len(df.query(self.icd_query("543,571,572")))*scale
    count_years["Cancer"] = len(df.query(self.icd_query("140-205")))*scale
    count_years["Diabetes"] = len(df.query(self.icd_query("260")))*scale
    count_years["Heart disease"] = len(df.query(self.icd_query("400-402,410-443")))*scale
    count_years["Stroke"] = len(df.query(self.icd_query("330-334")))*scale
    count_years["Arteriosclerosis"] = len(df.query(self.icd_query("450")))*scale
    count_years["Influenza and pneumonia"] = len(df.query(self.icd_query("480-493")))*scale
    count_years["Cirrhosis of liver"] = len(df.query(self.icd_query("581")))*scale
    count_years["Kidney disease"] = len(df.query(self.icd_query("590-594")))*scale
    count_years["Congenital Malformations"] = len(df.query(self.icd_query("750-759")))*scale
    count_years["Perinatal Conditions"] = len(df.query(self.icd_query("760-776")))*scale
    count_years["Motor vehicle accidents"] = len(df.query(self.icd_query("810-835")))*scale
    count_years["Accidents excluding motor vehicles"] = len(df.query(self.icd_query("800-802,840-962")))*scale
    count_years["Typhoid fever"] = len(df.query(self.icd_query("040")))*scale
    count_years["Measles"] = len(df.query(self.icd_query("085")))*scale
    count_years["Whooping cough"] = len(df.query(self.icd_query("056")))*scale
    count_years["Diphtheria"] = len(df.query(self.icd_query("055")))*scale
    count_years["Intestinal infections"] = len(df.query(self.icd_query("571,764")))*scale
    count_years["Meningococcal infections"] = len(df.query(self.icd_query("057")))*scale
    count_years["Acute poliomyelitis"] = len(df.query(self.icd_query("080")))*scale
    count_years["Syphilis"] = len(df.query(self.icd_query("020-029")))*scale
    count_years["Acute rheumatic fever"] = len(df.query(self.icd_query("400-402")))*scale
    count_years["Hypertension"] = len(df.query(self.icd_query("444-447")))*scale
    count_years["Chronic respiratory diseases"] = len(df.query(self.icd_query("241,501,502,527.1")))*scale
    count_years["Ulcer"] = len(df.query(self.icd_query("540,541")))*scale
    count_years["Suicide"] = len(df.query(self.icd_query("963,970-979")))*scale
    count_years["Homicide"] = len(df.query(self.icd_query("964,980-985")))*scale
  
  def longterm_comparable_icd8(self, df, count_years, scale, comparability_ratios):
    count_years["Tuberculosis"] = len(df.query(self.icd_query("010-019")))*scale
    count_years["Diarrhea, enteritis, and colitis"] = len(df.query(self.icd_query("009")))*scale
    count_years["Cancer"] = len(df.query(self.icd_query("140-209")))*scale
    count_years["Diabetes"] = len(df.query(self.icd_query("250")))*scale
    count_years["Heart disease"] = len(df.query(self.icd_query("390-398,402,404,410-429")))*scale
    count_years["Stroke"] = len(df.query(self.icd_query("430-438")))*scale
    count_years["Arteriosclerosis"] = len(df.query(self.icd_query("440")))*scale
    count_years["Influenza and pneumonia"] = len(df.query(self.icd_query("470-474,480-486")))*scale
    count_years["Cirrhosis of liver"] = len(df.query(self.icd_query("571")))*scale
    count_years["Kidney disease"] = len(df.query(self.icd_query("580-584")))*scale
    count_years["Congenital Malformations"] = len(df.query(self.icd_query("740-759")))*scale
    count_years["Perinatal Conditions"] = len(df.query(self.icd_query("760-769.2,769.4-772,774-778")))*scale
    count_years["Motor vehicle accidents"] = len(df.query(self.icd_query("810-823")))*scale
    count_years["Accidents excluding motor vehicles"] = len(df.query(self.icd_query("800-807,825-949")))*scale
    count_years["Typhoid fever"] = len(df.query(self.icd_query("001")))*scale
    count_years["Measles"] = len(df.query(self.icd_query("055")))*scale
    count_years["Whooping cough"] = len(df.query(self.icd_query("033")))*scale
    count_years["Diphtheria"] = len(df.query(self.icd_query("032")))*scale
    count_years["Intestinal infections"] = len(df.query(self.icd_query("008,009")))*scale
    count_years["Meningococcal infections"] = len(df.query(self.icd_query("036")))*scale
    count_years["Acute poliomyelitis"] = len(df.query(self.icd_query("040-043")))*scale
    count_years["Syphilis"] = len(df.query(self.icd_query("090-097")))*scale
    count_years["Acute rheumatic fever"] = len(df.query(self.icd_query("390-392")))*scale
    count_years["Hypertension"] = len(df.query(self.icd_query("400,401,403")))*scale
    count_years["Chronic respiratory diseases"] = len(df.query(self.icd_query("490-493")))*scale
    count_years["Ulcer"] = len(df.query(self.icd_query("531-533")))*scale
    count_years["Suicide"] = len(df.query(self.icd_query("950-959")))*scale
    count_years["Homicide"] = len(df.query(self.icd_query("960-978")))*scale

  def longterm_comparable_icd9(self, df, count_years, scale, comparability_ratios):
    count_years["Tuberculosis"] = len(df.query(self.icd_query("010-018")))*scale
    count_years["Diarrhea, enteritis, and colitis"] = len(df.query(self.icd_query("009")))*scale
    count_years["Cancer"] = len(df.query(self.icd_query("140-208")))*scale
    count_years["Diabetes"] = len(df.query(self.icd_query("250")))*scale
    count_years["Heart disease"] = len(df.query(self.icd_query("390-398,402,404,410-429")))*scale
    count_years["Stroke"] = len(df.query(self.icd_query("430-438")))*scale
    count_years["Arteriosclerosis"] = len(df.query(self.icd_query("440")))*scale
    count_years["Influenza and pneumonia"] = len(df.query(self.icd_query("480-487")))*scale
    count_years["Cirrhosis of liver"] = len(df.query(self.icd_query("571")))*scale
    count_years["Kidney disease"] = len(df.query(self.icd_query("580-589")))*scale
    count_years["Congenital Malformations"] = len(df.query(self.icd_query("740-759")))*scale
    count_years["Perinatal Conditions"] = len(df.query(self.icd_query("760-779")))*scale
    count_years["Motor vehicle accidents"] = len(df.query(self.icd_query("810-825")))*scale
    count_years["Accidents excluding motor vehicles"] = len(df.query(self.icd_query("800-807,826-949")))*scale
    count_years["Typhoid fever"] = len(df.query(self.icd_query("002.0")))*scale
    count_years["Measles"] = len(df.query(self.icd_query("055")))*scale
    count_years["Whooping cough"] = len(df.query(self.icd_query("033")))*scale
    count_years["Diphtheria"] = len(df.query(self.icd_query("032")))*scale
    count_years["Intestinal infections"] = len(df.query(self.icd_query("007-009")))*scale
    count_years["Meningococcal infections"] = len(df.query(self.icd_query("036")))*scale
    count_years["Acute poliomyelitis"] = len(df.query(self.icd_query("045")))*scale
    count_years["Syphilis"] = len(df.query(self.icd_query("090-097")))*scale
    count_years["Acute rheumatic fever"] = len(df.query(self.icd_query("390-392")))*scale
    count_years["Hypertension"] = len(df.query(self.icd_query("401,403")))*scale
    count_years["Chronic respiratory diseases"] = len(df.query(self.icd_query("490-496")))*scale
    count_years["Ulcer"] = len(df.query(self.icd_query("531-533")))*scale
    count_years["Suicide"] = len(df.query(self.icd_query("950-959")))*scale
    count_years["Homicide"] = len(df.query(self.icd_query("960-978")))*scale

  def longterm_comparable_icd10(self, df, count_years, scale, comparability_ratios):
    count_years["Tuberculosis"] = len(df.query(self.icd_query("A16-A19")))*scale
    count_years["Diarrhea, enteritis, and colitis"] = len(df.query(self.icd_query("A09")))*scale
    count_years["Cancer"] = len(df.query(self.icd_query("C00-C97")))*scale
    count_years["Diabetes"] = len(df.query(self.icd_query("E10-E14")))*scale
    count_years["Heart disease"] = len(df.query(self.icd_query("I00-I09,I11,I13,I20-I51")))*scale
    count_years["Stroke"] = len(df.query(self.icd_query("I60-I69,G45")))*scale
    count_years["Arteriosclerosis"] = len(df.query(self.icd_query("I70")))*scale
    count_years["Influenza and pneumonia"] = len(df.query(self.icd_query("J10-J18")))*scale
    count_years["Cirrhosis of liver"] = len(df.query(self.icd_query("K70,K73-K74")))*scale
    count_years["Kidney disease"] = len(df.query(self.icd_query("N00-N07,N17-N19,N25-N27")))*scale
    count_years["Congenital Malformations"] = len(df.query(self.icd_query("Q00-Q99")))*scale
    count_years["Perinatal Conditions"] = len(df.query(self.icd_query("P00-P96, A33")))*scale
    count_years["Motor vehicle accidents"] = len(df.query(self.icd_query("V02-V04,V09.0,V09.2,V12-V14,V19.0-V19.2,V19.4-V19.6,V20-V79,V80.3-V80.5,V81.0-V81.1,V82.0-V82.1,V83-V86,V87.0-V87.8,V88.0-V88.8,V89.0,V89.2")))*scale
    count_years["Accidents excluding motor vehicles"] = len(df.query(self.icd_query("V01,V05-V08,V09.1,V09.3-V11,V15-V18,V19.3,V19.7-V19.9,V80.0-V80.2,V80.6-V80.9,V81.2-V81.9,V82.2-V82.9,V87.9,V88.9,V89.1,V89.3-X59,Y85-Y86")))*scale
    count_years["Typhoid fever"] = len(df.query(self.icd_query("A01.0")))*scale
    count_years["Measles"] = len(df.query(self.icd_query("B05")))*scale
    count_years["Whooping cough"] = len(df.query(self.icd_query("A37")))*scale
    count_years["Diphtheria"] = len(df.query(self.icd_query("A36")))*scale
    count_years["Intestinal infections"] = len(df.query(self.icd_query("A04,A07-A09")))*scale
    count_years["Meningococcal infections"] = len(df.query(self.icd_query("A39")))*scale
    count_years["Acute poliomyelitis"] = len(df.query(self.icd_query("A80")))*scale
    count_years["Syphilis"] = len(df.query(self.icd_query("A50-A53")))*scale
    count_years["Acute rheumatic fever"] = len(df.query(self.icd_query("I00-I02")))*scale
    count_years["Hypertension"] = len(df.query(self.icd_query("I10, I12")))*scale
    count_years["Chronic respiratory diseases"] = len(df.query(self.icd_query("J40-J47,J67")))*scale
    count_years["Ulcer"] = len(df.query(self.icd_query("K25-K28")))*scale
    count_years["Suicide"] = len(df.query(self.icd_query("X60-X84,Y87.0")))*scale
    count_years["Homicide"] = len(df.query(self.icd_query("X85-Y09,Y35,Y87.1,Y89.0")))*scale

  def get_raw_mortality_counts(self, yearly_basis, process_icd7, process_icd8, process_icd9, process_icd10, min_year=None, calculate_total=True, add_icd_revision=True, comparability_ratios=None):
    counts = {}
    csvs = self.get_mortality_files()
    for i, csv in enumerate(csvs):
      filename, file_extension, file_year = self.get_mortality_file_info(csv)
      if filename:
        if min_year is not None and file_year < min_year:
          continue
        count_years = yearly_basis()
        counts[file_year] = count_years

        if add_icd_revision:
          count_years["ICD Revision"] = UCODUnitedStates.mortality_uspopulation.loc[file_year]["ICDRevision"]

        self.print_processing_csv(i, csv, csvs)
        
        df, scale = self.get_mortality_data(csv, file_year)
        
        if calculate_total:
          count_years["Total Deaths"] = len(df)*scale

        if file_year >= 1958 and file_year <= 1967:
          process_icd7(df, count_years, scale, comparability_ratios)
        elif file_year >= 1968 and file_year <= 1978:
          process_icd8(df, count_years, scale, comparability_ratios)
        elif file_year >= 1979 and file_year <= 1998:
          process_icd9(df, count_years, scale, comparability_ratios)
        elif file_year >= 1999:
          process_icd10(df, count_years, scale, comparability_ratios)

    df = pandas.DataFrame.from_dict(counts, orient="index")
    return df
    
  def process_raw_mortality_data(self):
    df = self.get_raw_mortality_counts(
      self.get_longterm_comparable_yearly_basis,
      self.longterm_comparable_icd7(),
      self.longterm_comparable_icd8(),
      self.longterm_comparable_icd9(),
      self.longterm_comparable_icd10(),
    )

    output_file = os.path.abspath(os.path.join(self.options.cachedir, "comparable_data_since_1959.xlsx"))
    df.to_excel(output_file)
    print("Created {}".format(output_file))

  def create_comparable(self):
    comparability_ratios = pandas.read_excel(self.options.data_comparable_ratios_input_file, sheet_name="Comparability Ratios", index_col=0, usecols=[0, 2, 4, 6, 8, 10]).fillna(1)
    comparable_ucods = pandas.read_excel(self.options.data_comparable_ratios_input_file, index_col="Year")
    comparable_ucods = comparable_ucods.transform(self.transform_row, axis="columns", comparability_ratios=comparability_ratios)
    comparable_ucods.to_excel(self.get_data_file())
    print("Created {}".format(self.get_data_file()))
    
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
    
  def get_calculated_scale_function_values(self):
    self.check_raw_files_directory()
    average_range = self.options.average_age_range
    min_year = self.data["Year"].max() - self.options.average_age_range + 1
    
    if self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS or \
       self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS or \
       self.options.data_type == DataType.US_ICD10_CHAPTERS:
      icd_codes = self.data[self.get_code_column_name()].unique()
      icd_codes_map = dict(zip(icd_codes, [{} for i in range(0, len(icd_codes))]))
    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:
      ulcl_codes = pandas.read_excel(
        self.options.data_comparable_ratios_input_file,
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
          if icd_range != "Residual" and icd_range != "NaN":
            ages = df.query(self.icd_query(icd_range))["AgeMinutes"]
            if type(ages) is numpy.float64:
              year_stats[icd_range] = {"Sum": ages, "Max": ages, "Count": 1, "Scale": scale}
            else:
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

    if self.options.data_type == DataType.US_ICD10_SUB_CHAPTERS or \
       self.options.data_type == DataType.US_ICD10_MINIMALLY_GROUPED or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ALL or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_LEAVES or \
       self.options.data_type == DataType.US_ICD_113_SELECTED_CAUSES_ROOTS or \
       self.options.data_type == DataType.US_ICD10_CHAPTERS:
      agesbygroup[self.obfuscated_column_name] = agesbygroup.apply(lambda row: self.get_obfuscated_name(self.data[self.data[self.get_code_column_name()] == row.name][self.get_action_column_name()].iloc[0]), raw=True, axis="columns")
    elif self.options.data_type == DataType.US_ICD_LONGTERM_COMPARABLE_LEADING:
      agesbygroup[self.obfuscated_column_name] = agesbygroup.apply(lambda row: self.get_obfuscated_name(ulcl_codes[ulcl_codes == row.name].index.values[0]), raw=True, axis="columns")
    else:
      raise NotImplementedError()

    self.write_spreadsheet(agesbygroup, self.prefix_all("agesbygroup"))
    result = agesbygroup[[self.obfuscated_column_name, calculated_col]]
    result.set_index(self.obfuscated_column_name, inplace=True)
    #result[self.options.average_ages] = vbp.normalize(result[self.options.average_ages].values, 0.5, 1.0)
    return result
  
  def get_icd9_10_comparability_ratios(self):
    df = pandas.read_excel(
      self.options.data_comparability_ratio_tables,
      header=None,
      skiprows=5,
      usecols="A:J",
      na_values=["*"],
      skipfooter=4,
    )
    df = df[[0, 9]]
    df.fillna(1, inplace=True) # Actual ratio is "Figure does not meet standards of reliability or precision"
    df.columns = ["List number", "Final comparability ratio"]
    return df
  
  def raw_icd9_counts(self, df, count_years, scale, comparability_ratios):
    for k,v in UCODUnitedStates.icd9_ucod113.recursive_dict(False).items():
      comparability_ratio = comparability_ratios[comparability_ratios["List number"] == k]["Final comparability ratio"].iloc[0]
      count_years[self.extract_name(v)] = len(df.query(self.icd_query(self.extract_codes(v))))*scale*comparability_ratio

  def raw_icd10_counts(self, df, count_years, scale, comparability_ratios):
    for k,v in UCODUnitedStates.icd10_ucod113.recursive_dict(False).items():
      count_years[self.extract_name(v)] = len(df.query(self.icd_query(self.extract_codes(v))))*scale

  def raw_icd_basis(self):
    x = {self.extract_name(i): numpy.NaN for i in UCODUnitedStates.icd10_ucod113.recursive_list(False)}
    return x
  
  def run_test(self):
    #self.ensure_loaded()
    return None
