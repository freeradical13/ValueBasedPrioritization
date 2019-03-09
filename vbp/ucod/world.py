import vbp
import sys
import enum
import numpy
import pandas
import datetime
import matplotlib

class UnderlyingCausesOfDeathWorld(vbp.DataSource):
  def run_load(self):
    populations = pandas.read_csv(
      "data/ucod/world/pop",
      usecols=["Country", "Year", "Pop1"],
    )
    
    populations.set_index(["Country", "Year"], inplace=True)

    future_unpopdata = self.read_unpopdata("MEDIUM VARIANT")
    
    # Drop 2015 because it's also in the ESTIMATES sheet so the join would fail
    future_unpopdata.drop(columns=["2015"], inplace=True)
    
    unpopdata = self.read_unpopdata("ESTIMATES").join(future_unpopdata)
    
    unpopdata.sort_index(inplace=True)
    
    self.write_spreadsheet(unpopdata, self.prefix_all("unpopdata"), use_digits_grouping_number_format=True)
    
    country_codes = pandas.read_csv(
      "data/ucod/world/country_codes",
      index_col=0,
      squeeze=True,
    )
    
    country_codes = country_codes.apply(str.strip)
    
    deaths = pandas.concat([
      self.read_icd10("data/ucod/world/Morticd10_part1"),
      self.read_icd10("data/ucod/world/Morticd10_part2"),
    ], sort=False, ignore_index=True)

    # Drop any with non-vanilla ICD-10 codes
    # https://www.who.int/healthinfo/statistics/documentation.zip
    deaths.drop(deaths[~deaths["List"].isin(["101", "103", "104"])].index, inplace=True)
    
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
    
    deaths = deaths.groupby(["Year", "Cause"]).agg({"Deaths": numpy.sum, "Population": numpy.sum})
    
    deaths["Crude Rate"] = (deaths["Deaths"] / deaths["Population"]) * 100000.0
    
    deaths.reset_index(level=deaths.index.names, inplace=True)

    self.write_spreadsheet(deaths, self.prefix_all("deaths"))
    
    return deaths

  def get_action_column_name(self):
    return "Cause"
  
  def get_value_column_name(self):
    return "Deaths"
  
  def run_predict(self):
    return self.prophet()

  def read_unpopdata(self, sheet_name):
    # https://population.un.org/wpp/Download/Standard/Population/
    # https://population.un.org/wpp/Publications/Files/WPP2017_Methodology.pdf (Pages 30-31)
    unpopdata = pandas.read_excel(
      "data/population/WPP2017_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx",
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
      },
    )
    df.rename(columns={"Deaths1": "Deaths"}, inplace=True)
    return df
