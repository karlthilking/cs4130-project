### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def load_datasets(dataset):
    data = load_dataset(dataset, split="train")
    df = data.to_pandas()
    df.columns = [column.lower() for column in df.columns]
    return df

def load_model_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_input(tokenizer, model, dataset_name: str, dataset, user_request, plot_type):
    df_columns = list(dataset.columns)
    condition_1 = f"You have a DataFrame called {dataset_name} with these columns: {df_columns}. Using this dataset, you will be asked to generate some plots for \
exploration purpose so the user can better understand the dataset. \nUser requirement: "

    condition_2 = f"\nBased on the requirement, generate one python function called Solution() without any parameters that generate the requested {plot_type}. When \
generating plots, use the Matplotlib library and the function shouldn't return anything but simply show the plot directly using plt.show(). Everything besides the \
function itself should not be included, so any comments, explanation, and reasoning are not needed."

    condition_3 = f"You have a DataFrame called {dataset_name} with these columns: {df_columns}. Here's the user requirement to explore the DataFrame: "

    condition_4 = "\nUse the .describe() function in pandas to get the key statistics information about the DataFrame. You should only print out the statistics of the \
columns involved in the user requirement, and write a brief summary paragraph of the plot described in the user requirement based on the statistics information. For \
example, given a health dataframe called health_df and the user request of 'Give me a scatter plot to show the relationship between height and weight', you should \
first use health_df.describe() and print out the statistics of column height and weight, strictly in the format of \
'{'height': {'mean': 10.2, 'std': 1.8, 'min': 1.2, ...}, 'weight': {'mean': 50.2, 'std': 2.8, 'min': 41.2, ...}}'.\
Then, under the header of 'Summary:', write a summary paragraph describing the pattern and trend of the scatter plot mentioned in the user requirement. \
\n\nYour statistics and summary paragraph:\n"

    code_prompt = condition_1 + user_request + condition_2
    stats_prompt = condition_3 + user_request + condition_4
    code_input_tokens = tokenizer(code_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    stats_input_tokens = tokenizer(stats_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    return code_input_tokens, stats_input_tokens

def generate_code(tokenizer, model, input_tokens):
    input_len = len(input_tokens.input_ids[0])
    model_output = model.generate(**input_tokens, max_new_tokens=300)
    output_code = tokenizer.decode(model_output[input_len:], skip_special_tokens=True)
    output_code = output_code.replace("```python```", "").replace("```", "")
    output_code = output_code.strip()
    return output_code

def generate_stats_summary(tokenizer, model, input_tokens):
    input_len = len(input_tokens.input_ids[0])
    model_output = model.generate(**input_tokens, max_new_tokens=300)
    output_texts = tokenizer.decode(model_output[input_len:], skip_special_tokens=True)
    output_texts = output_texts.strip()
    return output_texts

def execute_code(code, dataset_name, df):
    local_vars = {}
    global_vars = {dataset_name: df, "plt": plt, "pd": pd, "np": np}
    try:
        exec(code, global_vars, local_vars)
        if "Solution" in local_vars:
            local_vars["Solution"]()
            return 1
        else:
            print ("Generated function isn't named as 'Solution'")
            return 0
    except Exception as e:
        print (f"Error executing the code: {e}")
        return 0

def main():
    ev_data = "UrvishAhir1/Electric-Vehicle-Specs-Dataset-2025"
    flower_data = "brjapon/iris"
    ev_df = load_datasets(ev_data)
    flower_df = load_datasets(flower_data)
    
    model_name = "Qwen/Qwen3-8B"
    model, tokenizer = load_model_tokenizer(model_name)

    dataset_name =  None
    dataset = None
    while dataset_name == None and dataset == None:
        dataset_choice = int(input("Are you exploring the electric vehicle dataset or flower dataset today? Enter 1 or 2 to indicate:\n"))
        if dataset_choice == 1:
            dataset_name = "ev_df"
            dataset = ev_df
        elif dataset_choice == 2:
            dataset_name = "flower_df"
            dataset = flower_df
    user_request = input("\nAny columns mentioned should be enclosed in quotations. Enter your request:\n")
    plot_type = input("\nEnter the type of plot you want:\n")

    code_input_tokens, stats_input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request, plot_type)
    output_code = generate_code(tokenizer, model, code_input_tokens)
    stats_texts = generate_stats_summary(tokenizer, model, stats_input_tokens)
    
    print ("\nThanks, your request is received. I'm currently generating code to complete your request.")
    print (f"\n\nCode generated to complete your request: \n{output_code}\n\n")
    result = execute_code(output_code, dataset_name, dataset)
    if result == 1:
        print ("Your requested plots are successfully generated.")
    else:
        print ("Your requested plots failed to generate due to above errors.")
    print ("\n\nHere's the statistics information of the features included, and a summary paragraph of the plot:\n")
    print (stats_texts)
    
if __name__ == "__main__":
    main()
