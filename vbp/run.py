import io
import os
import vbp
import sys
import numpy
import pandas
import argparse
import traceback
import matplotlib
import matplotlib.pyplot

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
  try:
    args = sys.argv[1:]
    
    data_sources = vbp.DataSource.__subclasses__()
    data_source_names = [data_source.__name__ for data_source in data_sources]
    data_source_classes = {data_source.__name__: data_source for data_source in data_sources}

    parser = vbp.create_parser()
    parser.add_argument("-v", "--version", action="version", version="{}".format(vbp.VERSION))
    
    subparsers = parser.add_subparsers(title="command", help="Run `command -h` for additional help", dest="command_name", required=True)
    
    subparser = subparsers.add_parser("modeled_value_based_prioritization", help="Run modeled value based prioritization")
    add_data_source_arg(subparser, data_source_names)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("predict", help="Generate predictions for an action")
    add_data_source_arg(subparser, data_source_names)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("list_actions", help="List unique actions")
    add_data_source_arg(subparser, data_source_names)
    add_output_arg(subparser)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("count_actions", help="Count unique actions")
    add_data_source_arg(subparser, data_source_names)
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("action_data", help="Print data for an action")
    add_data_source_arg(subparser, data_source_names)
    subparser.add_argument("action", help="Action")
    add_remainder_arg(subparser)
    
    subparser = subparsers.add_parser("manual_scale_functions", help="Generate scale functions table")
    add_data_source_arg(subparser, data_source_names)
    add_output_arg(subparser)
    subparser.add_argument("-t", "--type", choices=["csv", "excel"], help="Output type", default="csv")
    subparser.add_argument("-p", "--prefix", help="Action prefix")
    subparser.add_argument("-s", "--suffix", help="Action suffix")
    subparser.add_argument("-n", "--sheet_name", help="Excel sheet name", default="Sheet1")
    subparser.add_argument("column", help="Column", nargs="+")
    add_remainder_arg(subparser)

    subparser = subparsers.add_parser("prophet", help="Run prophet")
    add_data_source_arg(subparser, data_source_names)
    subparser.add_argument("action", help="Action")
    add_remainder_arg(subparser)
    
    options = parser.parse_args(args)
    if options.command_name == "modeled_value_based_prioritization":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      b = ds.modeled_value_based_prioritization()
      vbp.print_full_columns(b)
    elif options.command_name == "predict":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      b = ds.predict()
      vbp.print_full_columns(b)
    elif options.command_name == "list_actions":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      options.output.write(os.linesep.join(numpy.sort(ds.get_possible_actions()).tolist()))
      options.output.write(os.linesep)
      options.output.close()
    elif options.command_name == "count_actions":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      print(len(ds.get_possible_actions()))
    elif options.command_name == "manual_scale_functions":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      
      actions = numpy.sort(ds.get_possible_actions())
      
      data = {ds.obfuscated_column_name: actions}
      
      for column in options.column:
        data[column] = numpy.ones(len(actions))
        
      df = pandas.DataFrame(data, range(1, len(actions) + 1))

      df.index.name = ds.action_number_column_name
      
      if options.prefix is not None or options.suffix is not None:
        df[ds.pretty_action_column_name] = df[ds.obfuscated_column_name]
        
        # Re-order columns
        df = df.reindex(columns=([ds.pretty_action_column_name] + list([x for x in df.columns if x != ds.pretty_action_column_name and x != ds.obfuscated_column_name] + [ds.obfuscated_column_name])))
        
        if options.prefix is not None:
          df[ds.pretty_action_column_name] = options.prefix + df[ds.pretty_action_column_name]
        
        if options.suffix is not None:
          df[ds.pretty_action_column_name] = df[ds.pretty_action_column_name] + options.suffix
      
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
    elif options.command_name == "action_data":
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      print(ds.get_action_data(options.action))
    elif options.command_name == "prophet":
      import fbprophet
      ds = create_data_source(data_source_classes, options)
      ds.load(options.args)
      df = ds.get_action_data(options.action)
      df.reset_index(inplace=True)
      df = df[["Date", "Crude Rate"]]
      df.rename(columns={"Date": "ds", "Crude Rate": "y"}, inplace=True)
      print(df)
      prophet = fbprophet.Prophet()
      prophet.fit(df)
      future = prophet.make_future_dataframe(periods=10, freq="Y")
      forecast = prophet.predict(future)

      prophet.plot(forecast)
      matplotlib.pyplot.show()
      
      prophet.plot_components(forecast)
      matplotlib.pyplot.show()
      
      forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
      print(forecast)
    else:
      raise NotImplementedError()
  except:
    e = sys.exc_info()[0]
    if e != SystemExit:
      traceback.print_exc()
