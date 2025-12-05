### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

import yaml
from project_2 import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.image import AxesImage 
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

def evaluate(code, df, columns, plot_type):
  eval = {
    'execution_success': True,
    'correct_num_functions': False,
    'function_names': False,
    'no_parameters': False,
    'no_extra_code': False,
    'correct_df_name': False,
    'correct_labels': False,
  }
  try:
    parsed_code = ast.parse(code)

    nodes_f = [
      n for n in parsed_code.body if isinstance(n, ast.FunctionDef)
    ]
    nodes_uf = [
      n for n in parsed_code.body
    ]
    df_columns = list(df.columns); columns_used = []

    if 'plot' and 'stats' in [n.name for n in nodes_f]:
      eval['function_names'] = True  
      eval['correct_num_functions'] = True

    if all([len(n.args.args) == 0 for n in nodes_f]):
      eval['no_parameters'] = True

    if all([isinstance(n, ast.FunctionDef) for n in nodes_uf]):
      eval['no_extra_code'] = True

    for n in ast.walk(parsed_code):
      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
        if n.id == 'df':
          eval['correct_df_name'] = True

    locals = {}
    globals = {
      'df': df,
      'plt': plt,
      'pd': pd,
      'np': np,
      '__builtins__': __builtins__
    }

    try:
      exec(code, globals, locals)
      if 'plot' in locals and callable(locals['plot']):
        plt.close('all')
        locals['plot']()
        ax = plt.gca()
        has_title = bool(ax.get_title)
        has_xlabel = bool(ax.get_xlabel())
        has_ylabel = bool(ax.get_ylabel())
        if has_title and has_xlabel and has_ylabel:
          eval['correct_labels'] = True
      
    except:
      eval['execution_success'] = False
  except:
    for v in eval.values():
      v = False 
  return eval

def correct_columns(code, columns):
  if code is None:
    return False
  try:
    found_columns = []
    for c in columns:
      if c.lower() in code.lower() and c.lower() not in found_columns:
        found_columns.append(c.lower())
  except:
    return False
  return columns.sort() == found_columns.sort()

def correct_stats(stats, columns):
  if stats is None:
    return False
  try:
    found_columns = []
    for s in stats.columns:
      for c in columns:
        if c.lower() in s.lower() and c.lower() not in found_columns:
          found_columns.append(c.lower())
  except:
    return False
  return found_columns.sort() == columns.sort()

def correct_plot_type(code, plot_type):
  bar = lambda code: 'bar' in code or 'barh' in code
  scatter = lambda code: 'scatter' in code 
  density = lambda code: 'plot' and 'kde' in code
  histogram = lambda code: 'hist' in code
  pie = lambda code: 'pie' in code 
  heatmap = lambda code: any(x in code.lower() for x in ['sns.heatmap', 'pcolormesh', 'imshow'])
  box = lambda code: 'boxplot' in code
  violin = lambda code: 'violinplot' in code
  line = lambda code: 'plt.plot' in code

  match plot_type:
    case 'Bar chart':
      return bar(code)
    case 'Scatter plot':
      return scatter(code)
    case 'Density plot':
      return density(code)
    case 'Histogram':
      return histogram(code)
    case 'Pie chart':
      return pie(code)
    case 'Heat map':
      return heatmap(code)
    case 'Box plot':
      return box(code)
    case 'Violin plot':
      return violin(code)
    case 'Line chart':
      return line(code)
    case _:
      return False

if __name__ == '__main__':
  model = Model()
  
  datasets = {
    'ev_df': model.csv_to_df('Data/electric_vehicles_spec_2025.csv'),
    'flower_df': model.csv_to_df('Data/Iris.csv')
  }

  with open('tests.yaml') as f:
    tests = yaml.safe_load(f)
  
  eval = {
    'Execution successful': 0,
    'Correct number of functions': 0,
    'Correct function name': 0,
    'Correct number of function parameters': 0, 
    'No extra reasoning/explanation/comments': 0, 
    'Correct DataFrame name used': 0,
    'Titles and x, y labels properly created': 0, 
    'Correct column names used': 0,
    'Correct dataset statistics': 0,
    'Correct plot type': 0
  }

  for test in tqdm(tests['test_cases'], total=len(tests['test_cases'])):
    df_name = test['dataset_name']
    request = test['user_request']
    df = datasets[df_name]
    columns = test['columns']
    plot_type = test['plot_type']
    code = model.generate_code(df, df_name, request)
    stats = model.execute_code(code, df)
    result = evaluate(code, df, columns, plot_type)

    if result['execution_success']:
      eval['Execution successful'] += 1
    if result['correct_num_functions']:
      eval['Correct number of functions'] += 1
    if result['function_names']:
      eval['Correct function name'] += 1
    if result['no_parameters']:
      eval['Correct number of function parameters'] += 1
    if result['no_extra_code']:
      eval['No extra reasoning/explanation/comments'] += 1
    if result['correct_df_name']:
      eval['Correct DataFrame name used'] += 1
    if result['correct_labels']:
      eval['Titles and x, y labels properly created'] += 1
    if correct_columns(code, columns):
      eval['Correct column names used'] += 1
    if correct_stats(stats, columns):
      eval['Correct dataset statistics'] += 1
    if correct_plot_type(code, plot_type):
      eval['Correct plot type'] += 1
  
  for k, v in eval.items():
    print(f'{k}: {v/50}')