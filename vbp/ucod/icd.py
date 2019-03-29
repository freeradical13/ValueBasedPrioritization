import vbp
import numpy

from vbp import DictTree

# https://en.wikipedia.org/wiki/International_Statistical_Classification_of_Diseases_and_Related_Health_Problems
class ICD:
  
  # https://apps.who.int/iris/bitstream/handle/10665/246208/9789241549165-V1-eng.pdf
  # Pages 33-95
  icd10_chapters_and_subchapters = DictTree({
    1: DictTree(value="Certain infectious and parasitic diseases (A00-B99)", d={
      2: "Intestinal infectious diseases (A00-A09)",
      3: "Tuberculosis (A15-A19)",
      4: "Certain zoonotic bacterial diseases (A20-A28)",
      5: "Other bacterial diseases (A30-A49)",
      6: "Infections with a predominantly sexual mode of transmission (A50-A64)",
      7: "Other spirochaetal diseases (A65-A69)",
      8: "Other diseases caused by chlamydiae (A70-A74)",
      9: "Rickettsioses (A75-A79)",
      10: "Viral infections of the central nervous system (A80-A89)",
      11: "Arthropod-borne viral fevers and viral haemorrhagic fevers (A92-A99)",
      12: "Viral infections characterized by skin and mucous membrane lesions (B00-B09)",
      13: "Viral hepatitis (B15-B19)",
      14: "Human immunodeficiency virus [HIV] disease (B20-B24)",
      15: "Other viral diseases (B25-B34)",
      16: "Mycoses (B35-B49)",
      17: "Protozoal diseases (B50-B64)",
      18: "Helminthiases (B65-B83)",
      19: "Pediculosis, acariasis and other infestations (B85-B89)",
      20: "Sequelae of infectious and parasitic diseases (B90-B94)",
      21: "Bacterial, viral and other infectious agents (B95-B98)",
      22: "Other infectious diseases (B99)",
    }),
    23: DictTree(value="Neoplasms (C00-D48)", d={
      24: "Malignant neoplasms (C00-C97)",
      25: "In situ neoplasms (D00-D09)",
      26: "Benign neoplasms (D10-D36)",
      27: "Neoplasms of uncertain or unknown behaviour (D37-D48)",
    }),
    28: DictTree(value="Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism (D50-D89)", d={
      29: "Nutritional anaemias (D50-D53)",
      30: "Haemolytic anaemias (D55-D59)",
      31: "Aplastic and other anaemias (D60-D64)",
      32: "Coagulation defects, purpura and other haemorrhagic conditions (D65-D69)",
      33: "Other diseases of blood and blood-forming organs (D70-D77)",
      34: "Certain disorders involving the immune mechanism (D80-D89)",
    }),
    35: DictTree(value="Endocrine, nutritional and metabolic diseases (E00-E90)", d={
      36: "Disorders of thyroid gland (E00-E07)",
      37: "Diabetes mellitus (E10-E14)",
      38: "Other disorders of glucose regulation and pancreatic internal secretion (E15-E16)",
      39: "Disorders of other endocrine glands (E20-E35)",
      40: "Malnutrition (E40-E46)",
      41: "Other nutritional deficiencies (E50-E64)",
      42: "Obesity and other hyperalimentation (E65-E68)",
      43: "Metabolic disorders (E70-E90)",
    }),
    44: DictTree(value="Mental and behavioural disorders (F00-F99)", d={
      45: "Organic, including symptomatic, mental disorders (F00-F09)",
      46: "Mental and behavioural disorders due to psychoactive substance use (F10-F19)",
      47: "Schizophrenia, schizotypal and delusional disorders (F20-F29)",
      48: "Mood [affective] disorders (F30-F39)",
      49: "Neurotic, stress-related and somatoform disorders (F40-F48)",
      50: "Behavioural syndromes associated with physiological disturbances and physical factors (F50-F59)",
      51: "Disorders of adult personality and behaviour (F60-F69)",
      52: "Mental retardation (F70-F79)",
      53: "Disorders of psychological development (F80-F89)",
      54: "Behavioural and emotional disorders with onset usually occurring in childhood and adolescence (F90-F98)",
      55: "Unspecified mental disorder (F99)",
    }),
    56: DictTree(value="Diseases of the nervous system (G00-G99)", d={
      57: "Inflammatory diseases of the central nervous system (G00-G09)",
      58: "Systemic atrophies primarily affecting the central nervous system (G10-G14)",
      59: "Extrapyramidal and movement disorders (G20-G26)",
      60: "Other degenerative diseases of the nervous system (G30-G32)",
      61: "Demyelinating diseases of the central nervous system (G35-G37)",
      62: "Episodic and paroxysmal disorders (G40-G47)",
      63: "Nerve, nerve root and plexus disorders (G50-G59)",
      64: "Polyneuropathies and other disorders of the peripheral nervous system (G60-G64)",
      65: "Diseases of myoneural junction and muscle (G70-G73)",
      66: "Cerebral palsy and other paralytic syndromes (G80-G83)",
      67: "Other disorders of the nervous system (G90-G99)",
    }),
    68: DictTree(value="Diseases of the eye and adnexa (H00-H59)", d={
      69: "Disorders of eyelid, lacrimal system and orbit (H00-H06)",
      70: "Disorders of conjunctiva (H10-H13)",
      71: "Disorders of sclera, cornea, iris and ciliary body (H15-H22)",
      72: "Disorders of lens (H25-H28)",
      73: "Disorders of choroid and retina (H30-H36)",
      74: "Glaucoma (H40-H42)",
      75: "Disorders of vitreous body and globe (H43-H45)",
      76: "Disorders of optic nerve and visual pathways (H46-H48)",
      77: "Disorders of ocular muscles, binocular movement, accommodation and refraction (H49-H52)",
      78: "Visual disturbances and blindness (H53-H54)",
      79: "Other disorders of eye and adnexa (H55-H59)",
    }),
    80: DictTree(value="Diseases of the ear and mastoid process (H60-H95)", d={
      81: "Diseases of external ear (H60-H62)",
      82: "Diseases of middle ear and mastoid (H65-H75)",
      83: "Diseases of inner ear (H80-H83)",
      84: "Other disorders of ear (H90-H95)",
    }),
    85: DictTree(value="Diseases of the circulatory system (I00-I99)", d={
      86: "Acute rheumatic fever (I00-I02)",
      87: "Chronic rheumatic heart diseases (I05-I09)",
      88: "Hypertensive diseases (I10-I15)",
      89: "Ischaemic heart diseases (I20-I25)",
      90: "Pulmonary heart disease and diseases of pulmonary circulation (I26-I28)",
      91: "Other forms of heart disease (I30-I52)",
      92: "Cerebrovascular diseases (I60-I69)",
      93: "Diseases of arteries, arterioles and capillaries (I70-I79)",
      94: "Diseases of veins, lymphatic vessels and lymph nodes, not elsewhere classified (I80-I89)",
      95: "Other and unspecified disorders of the circulatory system (I95-I99)",
    }),
    96: DictTree(value="Diseases of the respiratory system (J00-J99)", d={
      97: "Acute upper respiratory infections (J00-J06)",
      98: "Influenza and pneumonia (J09-J18)",
      99: "Other acute lower respiratory infections (J20-J22)",
      100: "Other diseases of upper respiratory tract (J30-J39)",
      101: "Chronic lower respiratory diseases (J40-J47)",
      102: "Lung diseases due to external agents (J60-J70)",
      103: "Other respiratory diseases principally affecting the interstitium (J80-J84)",
      104: "Suppurative and necrotic conditions of lower respiratory tract (J85-J86)",
      105: "Other diseases of pleura (J90-J94)",
      106: "Other diseases of the respiratory system (J95-J99)",
    }),
    107: DictTree(value="Diseases of the digestive system (K00-K93)", d={
      108: "Diseases of oral cavity, salivary glands and jaws (K00-K14)",
      109: "Diseases of oesophagus, stomach and duodenum (K20-K31)",
      110: "Diseases of appendix (K35-K38)",
      111: "Hernia (K40-K46)",
      112: "Noninfective enteritis and colitis (K50-K52)",
      113: "Other diseases of intestines (K55-K64)",
      114: "Diseases of peritoneum (K65-K67)",
      115: "Diseases of liver (K70-K77)",
      116: "Disorders of gallbladder, biliary tract and pancreas (K80-K87)",
      117: "Other diseases of the digestive system (K90-K93)",
    }),
    118: DictTree(value="Diseases of the skin and subcutaneous tissue (L00-L99)", d={
      119: "Infections of the skin and subcutaneous tissue (L00-L08)",
      120: "Bullous disorders (L10-L14)",
      121: "Dermatitis and eczema (L20-L30)",
      122: "Papulosquamous disorders (L40-L45)",
      123: "Urticaria and erythema (L50-L54)",
      124: "Radiation-related disorders of the skin and subcutaneous tissue (L55-L59)",
      125: "Disorders of skin appendages (L60-L75)",
      126: "Other disorders of the skin and subcutaneous tissue (L80-L99)",
    }),
    127: DictTree(value="Diseases of the musculoskeletal system and connective tissue (M00-M99)", d={
      128: "Arthropathies (M00-M25)",
      129: "Systemic connective tissue disorders (M30-M36)",
      130: "Dorsopathies (M40-M54)",
      131: "Soft tissue disorders (M60-M79)",
      132: "Osteopathies and chondropathies (M80-M94)",
      133: "Other disorders of the musculoskeletal system and connective tissue (M95-M99)",
    }),
    134: DictTree(value="Diseases of the genitourinary system (N00-N99)", d={
      135: "Glomerular diseases (N00-N08)",
      136: "Renal tubulo-interstitial diseases (N10-N16)",
      137: "Renal failure (N17-N19)",
      138: "Urolithiasis (N20-N23)",
      139: "Other disorders of kidney and ureter (N25-N29)",
      140: "Other diseases of urinary system (N30-N39)",
      141: "Diseases of male genital organs (N40-N51)",
      142: "Disorders of breast (N60-N64)",
      143: "Inflammatory diseases of female pelvic organs (N70-N77)",
      144: "Noninflammatory disorders of female genital tract (N80-N98)",
      145: "Other disorders of the genitourinary system (N99)",
    }),
    146: DictTree(value="Pregnancy, childbirth and the puerperium (O00-O99)", d={
      147: "Pregnancy with abortive outcome (O00-O08)",
      148: "Oedema, proteinuria and hypertensive disorders in pregnancy, childbirth and the puerperium (O10-O16)",
      149: "Other maternal disorders predominantly related to pregnancy (O20-O29)",
      150: "Maternal care related to the fetus and amniotic cavity and possible delivery problems (O30-O48)",
      151: "Complications of labour and delivery (O60-O75)",
      152: "Delivery (O80-O84)",
      153: "Complications predominantly related to the puerperium (O85-O92)",
      154: "Other obstetric conditions, not elsewhere classified (O94-O99)",
    }),
    155: DictTree(value="Certain conditions originating in the perinatal period (P00-P96)", d={
      156: "Fetus and newborn affected by maternal factors and by complications of pregnancy, labour and delivery (P00-P04)",
      157: "Disorders related to length of gestation and fetal growth (P05-P08)",
      158: "Birth trauma (P10-P15)",
      159: "Respiratory and cardiovascular disorders specific to the perinatal period (P20-P29)",
      160: "Infections specific to the perinatal period (P35-P39)",
      161: "Haemorrhagic and haematological disorders of fetus and newborn (P50-P61)",
      162: "Transitory endocrine and metabolic disorders specific to fetus and newborn (P70-P74)",
      163: "Digestive system disorders of fetus and newborn (P75-P78)",
      164: "Conditions involving the integument and temperature regulation of fetus and newborn (P80-P83)",
      165: "Other disorders originating in the perinatal period (P90-P96)",
    }),
    167: DictTree(value="Congenital malformations, deformations and chromosomal abnormalities (Q00-Q99)", d={
      168: "Congenital malformations of the nervous system (Q00-Q07)",
      169: "Congenital malformations of eye, ear, face and neck (Q10-Q18)",
      170: "Congenital malformations of the circulatory system (Q20-Q28)",
      171: "Congenital malformations of the respiratory system (Q30-Q34)",
      172: "Cleft lip and cleft palate (Q35-Q37)",
      173: "Other congenital malformations of the digestive system (Q38-Q45)",
      174: "Congenital malformations of genital organs (Q50-Q56)",
      175: "Congenital malformations of the urinary system (Q60-Q64)",
      176: "Congenital malformations and deformations of the musculoskeletal system (Q65-Q79)",
      177: "Other congenital malformations (Q80-Q89)",
      178: "Chromosomal abnormalities, not elsewhere classified (Q90-Q99)",
    }),
    179: DictTree(value="Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)", d={
      180: "Symptoms and signs involving the circulatory and respiratory systems (R00-R09)",
      181: "Symptoms and signs involving the digestive system and abdomen (R10-R19)",
      182: "Symptoms and signs involving the skin and subcutaneous tissue (R20-R23)",
      183: "Symptoms and signs involving the nervous and musculoskeletal systems (R25-R29)",
      184: "Symptoms and signs involving the urinary system (R30-R39)",
      185: "Symptoms and signs involving cognition, perception, emotional state and behaviour (R40-R46)",
      186: "Symptoms and signs involving speech and voice (R47-R49)",
      187: "General symptoms and signs (R50-R69)",
      188: "Abnormal findings on examination of blood, without diagnosis (R70-R79)",
      189: "Abnormal findings on examination of urine, without diagnosis (R80-R82)",
      190: "Abnormal findings on examination of other body fluids, substances and tissues, without diagnosis (R83-R89)",
      191: "Abnormal findings on diagnostic imaging and in function studies, without diagnosis (R90-R94)",
      192: "Ill-defined and unknown causes of mortality (R95-R99)",
    }),
    193: DictTree(value="Injury, poisoning and certain other consequences of external causes (S00-T98)", d={
      194: "Injuries to the head (S00-S09)",
      195: "Injuries to the neck (S10-S19)",
      196: "Injuries to the thorax (S20-S29)",
      197: "Injuries to the abdomen, lower back, lumbar spine and pelvis (S30-S39)",
      198: "Injuries to the shoulder and upper arm (S40-S49)",
      199: "Injuries to the elbow and forearm (S50-S59)",
      200: "Injuries to the wrist and hand (S60-S69)",
      201: "Injuries to the hip and thigh (S70-S79)",
      202: "Injuries to the knee and lower leg (S80-S89)",
      203: "Injuries to the ankle and foot (S90-S99)",
      204: "Injuries involving multiple body regions (T00-T07)",
      205: "Injuries to unspecified part of trunk, limb or body region (T08-T14)",
      206: "Effects of foreign body entering through natural orifice (T15-T19)",
      207: "Burns and corrosions (T20-T32)",
      208: "Frostbite (T33-T35)",
      209: "Poisoning by drugs, medicaments and biological substances (T36-T50)",
      210: "Toxic effects of substances chiefly nonmedicinal as to source (T51-T65)",
      211: "Other and unspecified effects of external causes (T66-T78)",
      212: "Certain early complications of trauma (T79)",
      213: "Complications of surgical and medical care, not elsewhere classified (T80-T88)",
      214: "Sequelae of injuries, of poisoning and of other consequences of external causes (T90-T98)",
    }),
    215: DictTree(value="External causes of morbidity and mortality (V01-Y98)", d={
      216: "Accidents (V01-X59)",
      217: "Intentional self-harm (X60-X84)",
      218: "Assault (X85-Y09)",
      219: "Event of undetermined intent (Y10-Y34)",
      220: "Legal intervention and operations of war (Y35-Y36)",
      221: "Complications of medical and surgical care (Y40-Y84)",
      222: "Sequelae of external causes of morbidity and mortality (Y85-Y89)",
      223: "Supplementary factors related to causes of morbidity and mortality classified elsewhere (Y90-Y98)",
    }),
    224: DictTree(value="Factors influencing health status and contact with health services (Z00-Z99)", d={
      225: "Persons encountering health services for examination and investigation (Z00-Z13)",
      226: "Persons with potential health hazards related to communicable diseases (Z20-Z29)",
      227: "Persons encountering health services in circumstances related to reproduction (Z30-Z39)",
      228: "Persons encountering health services for specific procedures and health care (Z40-Z54)",
      229: "Persons with potential health hazards related to socioeconomic and psychosocial circumstances (Z55-Z65)",
      230: "Persons encountering health services in other circumstances (Z70-Z76)",
      231: "Persons with potential health hazards related to family and personal history and certain conditions influencing health status (Z80-Z99)",
    }),
    232: DictTree(value="Codes for special purposes (U00-U99)", d={
      233: "Provisional assignment of new diseases of uncertain etiology or emergency use (U00-U49)",
      234: "Resistance to antimicrobial and antineoplastic drugs (U82-U85)",
    }),
  })
  
  @staticmethod
  def toint(x, addone=False):
    x = x.replace("-", "")
    if len(x) == 4:
      x = x[0:3]
    normalized_x = ICD.normalize(x)
    try:
      result = int(normalized_x)
    except ValueError:
      #print("WARNING: Invalid ICD code {}. Interpreting first letter only.".format(x))
      #return ICD.blockint(x[0])
      return numpy.NaN
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
    normalized_x = ICD.normalize(x)
    try:
      return float(normalized_x) + f
    except ValueError:
      #print("WARNING: Invalid ICD code {}. Interpreting first letter only.".format(x))
      #return float(ICD.blockint(x[0])) + f
      return numpy.NaN

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
      if x[1:] == "999": # Age unknown
        return numpy.NaN

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

class ICDDataSource(vbp.TimeSeriesDataSource):
  def extract_codes(self, x):
    return x.strip()[x.strip().rfind("(")+1:][:-1]
  
  def extract_name(self, x):
    return x[:x.rfind("(")-1].replace("#", "").strip()

  def icd_query(self, codes):
    result = ""
    
    if codes != "Residual" and codes != "NaN":
      atol = 1e-08
      codes = codes.replace("*", "")
      codes_pieces = codes.split(",")
      for codes_piece in codes_pieces:
        if len(result) > 0:
          result += " | "
        result += "("
        codes_piece = codes_piece.strip()
        if "-" in codes_piece:
          range_pieces = codes_piece.split("-")
          x = range_pieces[0].strip()
          y = range_pieces[1].strip()
          if "." in x:
            floatval = ICD.tofloat(x)
            result += "(icdfloat >= {})".format(floatval - atol)
          else:
            result += "(icdint >= {})".format(ICD.toint(x))
          result += " & "
          if "." in y:
            floatval = ICD.tofloat(y)
            result += "(icdfloat <= {})".format(floatval + atol)
          else:
            result += "(icdint <= {})".format(ICD.toint(y))
        else:
          if "." in codes_piece:
            floatval = ICD.tofloat(codes_piece)
            result += "(icdfloat >= {}) & (icdfloat <= {})".format(floatval - atol, floatval + atol)
          else:
            result += "icdint == {}".format(ICD.toint(codes_piece))
        result += ")"
    else:
      result = "False"

    return result

  def crude_rate_amount(self):
    return 100000.0

  def get_value_column_name(self):
    return " (per {:,d})".format(int(self.crude_rate_amount()))
