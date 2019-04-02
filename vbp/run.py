import io
import os
import vbp
import sys
import numpy
import pandas
import inspect
import argparse
import traceback
import matplotlib
import matplotlib.pyplot

# Import all subclasses of DataSource so that they are options
import vbp.example
import vbp.ucod.united_states
import vbp.ucod.world

def add_data_source_arg(parser, data_source_names):
  parser.add_argument("data_source", choices=data_source_names, help="Data source")

def add_remainder_arg(parser):
  parser.add_argument("args", help="Arguments for the data source", nargs=argparse.REMAINDER)

def add_output_arg(parser):
  parser.add_argument("-o", "--output", help="Output file", type=argparse.FileType("wb"), default="-")

def add_all_arg(parser):
  parser.add_argument("-a", "--all", action="store_true", help="Perform command for all types", default=False)

def get_data_source(data_source_classes, options):
  return data_source_classes[options.data_source]

def create_data_source(data_source_classes, options):
  return data_source_classes[options.data_source]()

def write_str(output, str):
  if isinstance(output, io.TextIOWrapper):
    output.write(str)
  else:
    output.write(str.encode())

def get_data_types(data_source_classes, options):
  dsc = get_data_source(data_source_classes, options)
  default = dsc.get_data_types_enum_default()
  if default is not None:
    default = default.name
  data_types = [default]
  if options.all:
    e = dsc.get_data_types_enum()
    if e is not None:
      data_types = list(map(lambda x: x.name, list(e)))
  return data_types

def get_options_with_data_type(parser, data_type, args, clean):
  cloned_args = args.copy()
  if "--data-type" not in args and data_type is not None:
    cloned_args.append("--data-type")
    cloned_args.append(data_type)

  options = parser.parse_args(cloned_args)

  if not clean:
    options.args.append("--do-not-clean")
  
  return options

def add_data_source_subclasses(array, cl):
  data_sources = vbp.DataSource.__subclasses__()
  subclasses = cl.__subclasses__()
  if subclasses is not None:
    for subclass in subclasses:
      if not inspect.isabstract(subclass):
        array.append(subclass)
      add_data_source_subclasses(array, subclass)

def print_finished_ds(options):
  if options is not None:
    print("")
    print("Any output files such as images and spreadsheets have been written to {}".format(os.path.abspath(options.output_dir)))

def run_vbp():
  try:
    args = sys.argv[1:]
    
    data_sources = []
    add_data_source_subclasses(data_sources, vbp.DataSource)
    data_source_names = [data_source.__name__ for data_source in data_sources]
    data_source_classes = {data_source.__name__: data_source for data_source in data_sources}

    parser = vbp.create_parser()
    parser.add_argument("-v", "--version", action="version", version="{}".format(vbp.VERSION))
    
    subparsers = parser.add_subparsers(title="command", help="Run `command -h` for additional help", dest="command_name", required=True)
    
    subparser = subparsers.add_parser("modeled_value_based_prioritization", help="Run modeled value based prioritization")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("predict", help="Generate predictions for an action")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("list", help="List unique actions")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    add_output_arg(subparser)
    subparser.add_argument("--no-sort", help="Do not sort", action="store_true", default=False)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("count", help="Count unique actions")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("action_data", help="Print data for an action")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    add_remainder_arg(subparser)
    subparser.add_argument("action", help="Action")
    
    subparser = subparsers.add_parser("manual_scale_functions", help="Generate scale functions table")
    add_data_source_arg(subparser, data_source_names)
    add_all_arg(subparser)
    subparser.add_argument("-n", "--sheet_name", help="Excel sheet name", default="Sheet1")
    subparser.add_argument("-o", "--output", help="Output file")
    subparser.add_argument("-p", "--prefix", help="Action prefix")
    subparser.add_argument("-s", "--suffix", help="Action suffix")
    subparser.add_argument("-t", "--type", choices=["csv", "excel"], help="Output type", default="csv")
    subparser.add_argument("column", help="Column", nargs="+")
    add_remainder_arg(subparser)

    subparser = subparsers.add_parser("test", help="Test")
    add_data_source_arg(subparser, data_source_names)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("prepare_data", help="Prepare data")
    add_data_source_arg(subparser, data_source_names)
    add_remainder_arg(subparser)
    
    print("vbp version {}".format(vbp.VERSION))

    options = parser.parse_args(args)
    if options.command_name == "modeled_value_based_prioritization":
      
      first_ds = None
      data_types = get_data_types(data_source_classes, options)
      first = True
      for i, data_type in enumerate(data_types):
        if len(data_types) > 1:
          if data_type is not None:
            if i > 0:
              print("")
            print("Data type {}:".format(data_type))
            print("")

        options = get_options_with_data_type(parser, data_type, args, first)
        ds = create_data_source(data_source_classes, options)
        ds.load(options.args)
        b = ds.modeled_value_based_prioritization()
        print("")
        vbp.print_full_columns(b)
        first = False
        first_ds = ds.options

      print_finished_ds(first_ds)
        
    elif options.command_name == "predict":
      
      data_types = get_data_types(data_source_classes, options)
      first_ds = None
      first = True
      for i, data_type in enumerate(data_types):
        if len(data_types) > 1:
          if data_type is not None:
            if i > 0:
              print("")
            print("Data type {}:".format(data_type))
            print("")

        options = get_options_with_data_type(parser, data_type, args, first)
        ds = create_data_source(data_source_classes, options)
        ds.load(options.args)
        b = ds.predict()
        vbp.print_full_columns(b)
        first = False
        first_ds = ds.options

      print_finished_ds(first_ds)
      
    elif options.command_name == "list":
      
      data_types = get_data_types(data_source_classes, options)

      first = True
      first_ds = None
      for i, data_type in enumerate(data_types):
        if len(data_types) > 1:
          if data_type is not None:
            if i > 0:
              options.output.write(os.linesep)
            options.output.write("Data type {}:".format(data_type))
            options.output.write(os.linesep)

        options = get_options_with_data_type(parser, data_type, args, first)
        ds = create_data_source(data_source_classes, options)
        ds.load(options.args)
        if options.no_sort:
          options.output.write(os.linesep.join(ds.get_possible_actions().tolist()))
        else:
          options.output.write(os.linesep.join(numpy.sort(ds.get_possible_actions()).tolist()))
        options.output.write(os.linesep)
        first = False
        first_ds = ds.options

      options.output.close()

      print_finished_ds(first_ds)
      
    elif options.command_name == "count":
      
      data_types = get_data_types(data_source_classes, options)
      first = True
      first_ds = None
      for i, data_type in enumerate(data_types):
        if len(data_types) > 1:
          if data_type is not None:
            if i > 0:
              print("")
            print("Data type {}:".format(data_type))
            print("")

        options = get_options_with_data_type(parser, data_type, args, first)
        ds = create_data_source(data_source_classes, options)
        ds.load(options.args)
        print(len(ds.get_possible_actions()))
        first = False
        first_ds = ds.options

      print_finished_ds(first_ds)
      
    elif options.command_name == "manual_scale_functions":
      
      if options.output is None:
        raise ValueError("-o/--output FILE is required")
      
      data_types = get_data_types(data_source_classes, options)
      first = True
      first_ds = None
      for i, data_type in enumerate(data_types):
        if len(data_types) > 1:
          if data_type is not None:
            if i > 0:
              print("")
            print("Data type {}:".format(data_type))
            print("")

        options = get_options_with_data_type(parser, data_type, args, first)
        ds = create_data_source(data_source_classes, options)
        ds.load(options.args)

        outputname = ds.get_manual_scales_file(options.output)

        actions = numpy.sort(ds.get_possible_actions())
        
        data = {ds.obfuscated_column_name: actions}
        
        for column in options.column:
          data[column] = numpy.ones(len(actions))
          
        df = pandas.DataFrame(data, range(1, len(actions) + 1))

        df.index.name = ds.action_number_column_name
        
        if options.prefix is not None or options.suffix is not None:
          df[ds.pretty_action_column_name] = df[ds.obfuscated_column_name]
          
          # Re-order columns so that pretty_action_column_name is first,
          # obfuscated_column_name is last, and all other columns are
          # in between.
          df = df.reindex(columns=([ds.pretty_action_column_name] + list([x for x in df.columns if x != ds.pretty_action_column_name and x != ds.obfuscated_column_name] + [ds.obfuscated_column_name])))
          
          if options.prefix is not None:
            df[ds.pretty_action_column_name] = options.prefix + df[ds.pretty_action_column_name]
          
          if options.suffix is not None:
            df[ds.pretty_action_column_name] = df[ds.pretty_action_column_name] + options.suffix
        
        if options.type == "csv":
          df.to_csv(options.outputname)
          print("Wrote {}".format(outputname))
        elif options.type == "excel":
          writer = pandas.ExcelWriter(outputname, engine="xlsxwriter")
          df.to_excel(writer, sheet_name=options.sheet_name)
          writer.save()
          print("Wrote {}".format(outputname))
        else:
          raise NotImplementedError()

        first = False
        first_ds = ds.options

      print_finished_ds(first_ds)
      
    elif options.command_name == "action_data":
      
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      print(ds.get_action_data(options.action))
      print_finished_ds(ds.options)
      
    elif options.command_name == "prepare_data":
      
      ds = create_data_source(data_source_classes, options)
      ds.ensure_options(options.args)
      ds.prepare_data()
      print_finished_ds(ds.options)
      
    elif options.command_name == "test":
      
      ds = create_data_source(data_source_classes, options)
      ds.ensure_options(options.args)
      ds.test()
      print_finished_ds(ds.options)
      
    else:
      raise NotImplementedError()
    
  except:
    e = sys.exc_info()[0]
    if e != SystemExit:
      traceback.print_exc()

if __name__ == "__main__":
  run_vbp()
