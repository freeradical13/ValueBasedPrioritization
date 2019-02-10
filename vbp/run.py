import os
import vbp
import sys
import numpy
import argparse

# Import all subclasses of DataSource so that they are options
import vbp.ucod.united_states

def add_data_source_arg(parser, data_source_names):
  parser.add_argument("data_source", choices=data_source_names, help="Data source")

def add_remainder_arg(parser):
  parser.add_argument("args", help="Arguments for the data source", nargs=argparse.REMAINDER)

def create_data_source(data_source_classes, options):
  return data_source_classes[options.data_source]()

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
  subparser.add_argument("-o", "--output", help="Output file", type=argparse.FileType("w"), default="-")
  subparser.add_argument("-t", "--type", choices=["csv"], help="Table output type", default="csv")
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
