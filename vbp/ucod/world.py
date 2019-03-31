import os
import vbp
import sys
import enum
import numpy
import pandas
import zipfile
import datetime
import matplotlib
import vbp.ucod.icd
import urllib.request

from vbp.ucod.icd import ICD

class DataType(vbp.DataSourceDataType):
  WORLD_ICD10_MINIMALLY_GROUPED = enum.auto()
  WORLD_ICD10_CHAPTERS_ALL = enum.auto()
  WORLD_ICD10_CHAPTER_ROOTS = enum.auto()
  WORLD_ICD10_SUB_CHAPTERS = enum.auto()

class UCODWorld(vbp.ucod.icd.ICDDataSource):
  def initialize_parser(self, parser):
    super().initialize_parser(parser)
    parser.add_argument("--data-world-who-country-codes", help="WHO World country codes file", default=os.path.join(self.get_data_dir(), "data/ucod/world/country_codes"))
    parser.add_argument("--data-world-un-population", help="U.N. World population data file", default=os.path.join(self.get_data_dir(), "data/population/WPP2017_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx"))
    parser.add_argument("--data-world-who-population", help="WHO World population data file", default=os.path.join(self.get_data_dir(), "data/ucod/world/pop"))
    parser.add_argument("--download", help="If no files in --raw-files-directory, download and extract", action="store_true", default=True)
    parser.add_argument("--min-year", help="The minimum year to use data from", type=int, default=1999) # The data before 1999 is spotty and throws off crude rates significantly
    parser.add_argument("--max-year", help="The maximum year to use data from because not all data is reported", type=int, default=2015)
    parser.add_argument("--raw-files-directory", help="directory with raw files", default=os.path.join(self.default_cache_dir, "world"))

  @staticmethod
  def get_data_types_enum():
    return DataType

  @staticmethod
  def get_data_types_enum_default():
    return DataType.WORLD_ICD10_MINIMALLY_GROUPED

  def load_who_populations(self):
    populations = pandas.read_csv(
      self.options.data_world_who_population,
      usecols=["Country", "Year", "Pop1"],
    )
    populations.set_index(["Country", "Year"], inplace=True)
    return populations

  def load_un_populations(self):
    future_unpopdata = self.read_unpopdata("MEDIUM VARIANT")
    
    # Drop 2015 because it's also in the ESTIMATES sheet so the join would fail
    future_unpopdata.drop(columns=["2015"], inplace=True)
    
    unpopdata = self.read_unpopdata("ESTIMATES").join(future_unpopdata)
    
    unpopdata.sort_index(inplace=True)
    
    return unpopdata

  def load_country_codes(self):
    country_codes = pandas.read_csv(
      self.options.data_world_who_country_codes,
      index_col=0,
      squeeze=True,
    )
    
    country_codes = country_codes.apply(str.strip)
    return country_codes

  def download_raw_files(self):
    print("Downloading raw files from https://www.who.int/healthinfo/statistics/mortality_rawdata/en/")
    if not os.path.exists(self.options.raw_files_directory):
      os.makedirs(self.options.raw_files_directory)
    
    for i in ["Morticd10_part1.zip", "Morticd10_part2.zip"]:
      print("Downloading {}...".format(i))
      downloaded_file = os.path.join(self.options.raw_files_directory, i)
      urllib.request.urlretrieve("https://www.who.int/healthinfo/statistics/{}".format(i), downloaded_file)
      with zipfile.ZipFile(downloaded_file, "r") as zfile:
        print("Unzipping {}".format(i))
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

  def load_deaths(self, populations, unpopdata, country_codes):
    
    print("Loading death data. This will take a while but it will be cached for the future...")
    
    deaths = pandas.concat([
      self.read_icd10(os.path.join(self.options.raw_files_directory, "Morticd10_part1")),
      self.read_icd10(os.path.join(self.options.raw_files_directory, "Morticd10_part2")),
    ], sort=False, ignore_index=True)

    # Drop any with non-vanilla ICD-10 codes
    # https://www.who.int/healthinfo/statistics/documentation.zip
    deaths.drop(deaths[~deaths["List"].isin(["101", "103", "104"])].index, inplace=True)

    if self.options.min_year > 0:
      deaths.drop(deaths[deaths["Year"] < self.options.min_year].index, inplace=True)

    # Drop everything after the last year of full data
    deaths.drop(deaths[deaths["Year"] > self.options.max_year].index, inplace=True)
    
    # Drop anything with an invalid ICD10 code
    deaths.drop(deaths[deaths["Cause"].str.startswith("1")].index, inplace=True)
    
    # "Except United Kingdom which include data for England and Wales, 
    #  Northern Ireland and Scotland which are also presented separately, 
    #  all the other data are mutually exclusive, for e.g. Martinique,
    #  Guadeloupe are not included in France. Care should be taken when
    #  treating data of former entities which split up or merged together
    #  to form new entities."
    # https://www.who.int/healthinfo/statistics/documentation.zip
    # Therefore, we drop:
    # 4310,"United Kingdom, England and Wales"
    # 4320,"United Kingdom, Northern Ireland"
    # 4330,"United Kingdom, Scotland"
    deaths.drop(deaths[deaths["Country"].isin([4310, 4320, 4330])].index, inplace=True)
    
    # Death data is split by sex, so we sum up after grouping:
    deaths = deaths.groupby(["Country", "Year", "Cause"]).agg({"Deaths": numpy.sum})
    
    deaths = pandas.DataFrame(deaths.to_records())
    
    deaths["Population"] = deaths.apply(lambda row: self.find_population(row["Country"], row["Year"], populations, unpopdata, country_codes), axis="columns")
    
    deaths["icdint"] = deaths[self.get_action_column_name()].apply(ICD.toint)
    deaths["icdfloat"] = deaths[self.get_action_column_name()].apply(ICD.tofloat)
    deaths.dropna(inplace=True)

    return deaths

  def run_load(self):
    
    self.check_raw_files_directory()
    
    populations = self.load_with_cache("who_populations", self.load_who_populations)

    unpopdata = self.load_with_cache("un_populations", self.load_un_populations)

    self.write_spreadsheet(unpopdata, self.prefix_all("unpopdata"), use_digits_grouping_number_format=True)
    
    country_codes = self.load_with_cache("who_country_codes", self.load_country_codes)
    
    deaths = self.load_with_cache("who_deaths", self.load_deaths, populations, unpopdata, country_codes)
    
    print("Processing input data...")
    
    # The Population column is deaths per country per year, and we
    # want total population per year (for countries that reported
    # mortality data):
    deaths_per_year = deaths.groupby(["Year", "Country"]).aggregate({"Population": numpy.max}).groupby("Year").sum()

    if self.options.data_type == DataType.WORLD_ICD10_MINIMALLY_GROUPED:
      deaths = deaths.groupby(["Year", "Cause"]).agg({"Deaths": numpy.sum})
      deaths.reset_index(level=deaths.index.names, inplace=True)
      deaths["Population"] = deaths["Year"].apply(lambda y: deaths_per_year.loc[y])
      deaths[self.get_value_column_name()] = (deaths["Deaths"] / deaths["Population"]) * self.crude_rate_amount()
      deaths["Date"] = deaths["Year"].apply(lambda year: pandas.datetime.strptime(str(year), "%Y"))
      deaths = deaths[["Date"] + [col for col in deaths if col != "Date"]]
    elif self.options.data_type == DataType.WORLD_ICD10_CHAPTERS_ALL:
      deaths = self.process_grouping(deaths, deaths_per_year, False, False)
    elif self.options.data_type == DataType.WORLD_ICD10_SUB_CHAPTERS:
      deaths = self.process_grouping(deaths, deaths_per_year, False, True)
    elif self.options.data_type == DataType.WORLD_ICD10_CHAPTER_ROOTS:
      deaths = self.process_grouping(deaths, deaths_per_year, True, False)

    self.write_spreadsheet(deaths, self.prefix_all("deaths"))
    
    return deaths

  def process_grouping(self, deaths, deaths_per_year, roots_only, leaves_only):
    if roots_only:
      target = ICD.icd10_chapters_and_subchapters.roots_list()
    else:
      target = ICD.icd10_chapters_and_subchapters.recursive_list(leaves_only)
    df = pandas.DataFrame(index=pandas.MultiIndex.from_product([target, deaths["Year"].unique()], names = [self.get_action_column_name(), "Year"])).reset_index().sort_values("Year")
    df["Query"] = df.apply(lambda row: "(Year == {}) & ({})".format(row["Year"], self.icd_query(self.extract_codes(row[self.get_action_column_name()]))), axis="columns")
    df["Deaths"] = df["Query"].apply(lambda x: deaths.query(x)["Deaths"].sum())
    df["Population"] = df["Year"].apply(lambda y: deaths_per_year.loc[y])
    df[self.get_value_column_name()] = (df["Deaths"] / df["Population"]) * self.crude_rate_amount()
    df["Date"] = df["Year"].apply(lambda year: pandas.datetime.strptime(str(year), "%Y"))
    deaths = df[["Date"] + [col for col in df if col != "Date" and col != "Query"]]
    
    return deaths

  def get_action_column_name(self):
    return "Cause"
  
  def get_value_column_name(self):
    return "Crude Rate" + super().get_value_column_name()
  
  def crude_rate_amount(self):
    return 1000000.0

  def read_unpopdata(self, sheet_name):
    # https://population.un.org/wpp/Download/Standard/Population/
    # https://population.un.org/wpp/Publications/Files/WPP2017_Methodology.pdf (Pages 30-31)
    unpopdata = pandas.read_excel(
      self.options.data_world_un_population,
      skiprows=range(0, 16),
      index_col=2,
      sheet_name=sheet_name,
    )
    
    unpopdata.drop(columns=["Index", "Variant", "Notes", "Country code"], inplace=True)
    
    unpopdata.index = unpopdata.index.map(self.map_un_country_to_who_country)
    
    unpopdata = unpopdata * 1000
    
    return unpopdata
  
  def map_un_country_to_who_country(self, c):
    # Replace the UN country name with the WHO country name
    return c.strip() \
            .replace("Réunion", "Reunion") \
            .replace("Bolivia (Plurinational State of)", "Bolivia") \
            .replace("Caribbean Netherlands", "Netherlands Antilles") \
            .replace("Saint Vincent and the Grenadines", "Saint Vincent and Grenadines") \
            .replace("United States Virgin Islands", "Virgin Islands (USA)") \
            .replace("Venezuela (Bolivarian Republic of)", "Venezuela") \
            .replace("Cabo Verde", "Cape Verde") \
            .replace("State of Palestine", "Occupied Palestinian Territory")
  
  def find_population(self, country_code, year, populations, unpopdata, country_codes):
    if (country_code, year) in populations.index:
      # Population might be split up by sex, so we sum it up
      return populations.loc[(country_code, year), "Pop1"].sum()
    else:
      country_name = country_codes.loc[country_code]
      if country_name not in unpopdata.index:
        if country_code == 1365:
          # The country of Rodrigues is currently party of Mauritius,
          # but there could be data for that country as well, so
          # to avoid double counting population and given how small
          # data for Rodrigues is, to reduce complexity, we just discard
          # this data (this is only for the case that we have to fall
          # back to UN data if there is no match for [Rodrigues, Year]
          # in the WHO data).
          return None
        raise ValueError("Could not find population data for {}, {}, {}".format(country_name, country_code, year))
      country_data = unpopdata.loc[country_name]
      if str(year) not in country_data.index:
        raise ValueError("Could not find year {} for {}: {}".format(year, country_name, country_data))
      return country_data.loc[str(year)]
    
  def read_icd10(self, file):
    df = pandas.read_csv(
      file,
      usecols=["Country", "Year", "List", "Cause", "Deaths1"],
      dtype={
        "List": str,
        "Cause": str,
        "Deaths1": numpy.float64,
      },
    )
    df.rename(columns={"Deaths1": "Deaths"}, inplace=True)
    return df

  def get_action_title_prefix(self):
    return "Deaths from "
  
  def run_test(self):
    self.ensure_loaded()
    return None
