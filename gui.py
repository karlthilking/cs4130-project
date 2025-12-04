import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from project_test import Model

class GUI:
  def __init__(self):
    self.model = Model()
    self.df = None 
    self.df_name = None
  
  def load_dataset(self, df_name):
    self.df_name = df_name
    self.df = self.model.load_dataset(df_name)
    return ' '.join(list(self.df.columns))
  
  def load_csv(self, csv_path, df_name):
    df = pd.read_csv(csv_path)
    self.df = df; self.df_name = df_name 
    return ' '.join(list(self.df.columns))
  
  def generate_response(self, request):
    if self.df_name is None:
      return None, None, "select a dataset first"

    code = self.model.generate_code(self.df, self.df_name, request)
    stats = self.model.execute_code(code, self.df)
    plot = plt.gcf()
    summary = self.model.generate_summary(self.df, self.df_name, stats, request)

    if plot is None or stats is None:
      return None, None, "generation failed"
    return plot, stats.to_html(), summary

  def run(self):
    with gr.Blocks(title='Data Visualization Agent') as demo:
      with gr.Row():
        with gr.Column(scale=1):
          datasets = gr.Dropdown(
            label='Select Dataset',
            choices=list(self.model.datasets.keys()),
            value=list(self.model.datasets.keys())[0]
          )
          dataset_info = gr.Textbox(
            label='Dataset Labels',
            lines=6,
            interactive=False,
          )
          request = gr.Textbox(
            label='Enter Request',
            placeholder='Ask anything...',
            lines=3
          )
          generate = gr.Button(
            'Generate response',
            variant='primary',
            size='lg'
          )
        with gr.Column(scale=2):
          plot = gr.Plot()
          stats = gr.HTML()
          summary = gr.Textbox(
            label='Summary',
            lines=4,
            interactive=False
          )
      datasets.change(
        fn=self.load_dataset,
        inputs=[datasets],
        outputs=[dataset_info]
      )
      generate.click(
        fn=self.generate_response,
        inputs=[request],
        outputs=[plot, stats, summary]
      )
    demo.launch(
      server_port=8080,
      show_error=True
    )

if __name__ == '__main__':
  gui = GUI()
  gui.run()

  




