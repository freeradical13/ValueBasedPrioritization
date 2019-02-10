import io
import os
import vbp
import sys
import numpy
import pandas
import argparse

# Import all subclasses of DataSource so that they are options
import vbp.ucod.united_states

def add_data_source_arg(parser, data_source_names):
  parser.add_argument("data_source", choices=data_source_names, help="Data source")

def add_remainder_arg(parser):
  parser.add_argument("args", help="Arguments for the data source", nargs=argparse.REMAINDER)

def add_output_arg(parser):
  parser.add_argument("-o", "--output", help="Output file", type=argparse.FileType("wb"), default="-")

def create_data_source(data_source_classes, options):
  return data_source_classes[options.data_source]()

def write_str(output, str):
  if isinstance(output, io.TextIOWrapper):
    output.write(str)
  else:
    output.write(str.encode())

if __name__ == "__main__":
  args = sys.argv[1:]
  
  data_sources = vbp.DataSource.__subclasses__()
  data_source_names = [data_source.__name__ for data_source in data_sources]
  data_source_classes = {data_source.__name__: data_source for data_source in data_sources}

  parser = vbp.create_parser()
  parser.add_argument("-v", "--version", action="version", version="{}".format(vbp.VERSION))
  
  subparsers = parser.add_subparsers(title="command", help="Run `command -h` for additional help", dest="command_name", required=True)
  
  subparser = subparsers.add_parser("predict", help="Generate predictions for an action")
  add_data_source_arg(subparser, data_source_names)
  add_remainder_arg(subparser)
  
  subparser = subparsers.add_parser("list_actions", help="List unique actions")
  add_data_source_arg(subparser, data_source_names)
  add_output_arg(subparser)
  add_remainder_arg(subparser)
  
  subparser = subparsers.add_parser("scale_functions_table", help="Generate scale functions table")
  add_data_source_arg(subparser, data_source_names)
  add_output_arg(subparser)
  subparser.add_argument("-t", "--type", choices=["csv", "excel"], help="Output type", default="csv")
  subparser.add_argument("-p", "--prefix", help="Action prefix")
  subparser.add_argument("-s", "--suffix", help="Action suffix")
  subparser.add_argument("-n", "--sheet_name", help="Excel sheet name", default="Sheet1")
  subparser.add_argument("column", help="Column", nargs="+")
  add_remainder_arg(subparser)

  options = parser.parse_args(args)
  if options.command_name == "predict":
    ds = create_data_source(data_source_classes, options)
    ds.load(options.args)
    ds.predict()
  elif options.command_name == "list_actions":
    ds = create_data_source(data_source_classes, options)
    ds.load(options.args)
    options.output.write(os.linesep.join(numpy.sort(ds.get_possible_actions()).tolist()))
    options.output.write(os.linesep)
    options.output.close()
  elif options.command_name == "scale_functions_table":
    ds = create_data_source(data_source_classes, options)
    ds.load(options.args)
    
    actions = numpy.sort(ds.get_possible_actions())
    
    data = {"Action": actions}
    
    for column in options.column:
      data[column] = numpy.ones(len(actions))
      
    df = pandas.DataFrame(data, range(1, len(actions) + 1))

    df.index.name = "ActionNumber"

    if options.prefix is not None:
      df["Action"] = options.prefix + df["Action"]
    
    if options.suffix is not None:
      df["Action"] = df["Action"] + options.suffix
    
    if options.type == "csv":
      write_str(options.output, df.to_csv())
    elif options.type == "excel":
      output = io.BytesIO()
      writer = pandas.ExcelWriter(output, engine="xlsxwriter")
      df.to_excel(writer, sheet_name=options.sheet_name)
      writer.save()
      xlsx_data = output.getvalue()
      options.output.write(xlsx_data)
    else:
      raise NotImplementedError()
    
    options.output.close()
  else:
    raise NotImplementedError()
