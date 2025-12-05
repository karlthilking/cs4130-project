from groq import Groq
from typing import Union, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Model:
  def __init__(self):
    self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    self.model_name = "llama-3.3-70b-versatile"
    self.datasets = {
      "Electric Vehicles": "Data/electric_vehicles_spec_2025.csv",
      "Iris Flowers": "Data/Iris.csv",
      "Wine": "Data/wine.csv",
      "Stocks": "Data/stockdata.csv",
      "Doge Coin": "Data/dogecoin.csv"
    }

  def csv_to_df(self, csv_path: str):
    df = pd.read_csv(csv_path)
    return df

  def load_dataset(self, df_name):
    csv_path = self.datasets.get(df_name, None)
    if csv_path is None:
      return None
    else:
      return pd.read_csv(csv_path)

  def generate_code(self, df, df_name, request):
    prompt = f'''Generate Python code for data visualization.

    Dataset name: {df_name}
    Dataset categories: {', '.join(list(df.columns))}
    User request: {request}

    Your task:
      1. Write a function, "def plot()", to generate a plot according to user request using matplotlib.
      2. Write a second function, "def stats()", that returns a pandas DataFrame including stats about the specific data categories the user is interested in.
      3. Use the variable "df" in your functions to access the {df_name} dataset which has the categories listed previously.
    
    Mandatory Rules:
      1. Do not import any modules or libraries. All necessary libraries are already imported globally.
      2. Do not create new data - use "df" to access all necessary data.
      3. Do not include parameters in either function - you have global access to all variables and modules.
      4. No comments or explanations, only write code.
      5. Do not explicitly call either of your functions to execute them after writing their implementations. The user will invoke each functions on their own.
      6. Do not use plt.show(). The user will display the graph on their own.

    Here is an example outline to follow:
    def plot():
      plt.plot(df['column_1'], df['column_2'])
      plt.xlabel('column_1')
      plt.ylabel('column_2')
    
    def stats():
      return pd.DataFrame({{
        'mean_column_1': [df['column_1'].mean()],
        'mean_column_2': [df['column_2'].mean()]
      }})
    
    Request: {request}
    Write your functions here:'''
    try:
      response = self.client.chat.completions.create(
        messages=[{
          "role": "user", "content": prompt
        }],
        model=self.model_name,
        max_tokens=350,
        temperature=0.0
      )
      code = response.choices[0].message.content.replace('```python', '').replace('```', '').strip()
      
      self.messages = [
        {"role": "user", "content": prompt},
        response.choices[0].message
      ]
      return code
    except Exception as e:
      print(f'Error: {e}')
  
  def execute_code(self, code, df):
    locals = {}
    globals = {
      'plt': plt,
      'pd': pd,
      'np': np,
      'sns': sns,
      'df': df,
      '__builtins__': __builtins__
    }
    try:
      exec(code, globals, locals)
      if 'plot' in locals and callable(locals['plot']):
        plt.close('all')
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        locals['plot']()
      else:
        print('Failed to find plot function.\n')
      if 'stats' in locals and callable(locals['stats']):
        stats = locals['stats']()
        if stats.empty:
          return None
        if len(list(stats.columns)) > 5:
          stats = stats.iloc[:, :5]
        return stats
      else:
        print('Failed to find stats function.\n')
        return None
    except SyntaxError as e:
      print(f'Syntax Error: {e}.\n')
    except Exception as e:
      print(f'Error: {e}.\n')
    
  def generate_summary(self, df, df_name, stats, request):
    prompt=f'''You just generated a plot and stats for the user, now follow up with an explanation.
    Here are the stats you should consider: {stats}
    Explain important trends, relationships, correlations, etc. related to the findings from the plot and stat calculations. Keep your response to a very brief paragraph, only highlighting the most important details in order to be concise. Here was the user request: {request}'''

    try:
      response = self.client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=self.model_name,
        max_tokens=200,
        temperature=0.2
      ).choices[0].message.content
      return response
    except Exception as e:
      print(f'Error during summary generation: {e}')
      return None
 


  
