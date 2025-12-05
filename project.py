### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/12/2025

from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import re

def load_datasets(dataset):
    df = pd.read_csv(dataset)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df.columns = [column.lower() for column in df.columns]
    return df

def load_model_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_input(tokenizer, model, dataset_name: str, dataset, user_request, plot_type):
    df_columns = list(dataset.columns)
    user_request = user_request.lower()
    columns_used = re.findall(r"[\"'](.*?)[\"']", user_request)
    code_prompt = f"""
    # Generate Python code only, DO NOT include any explanations or comments.
    # You have a DataFrame called {dataset_name} with columns: {df_columns}
    # The function should be named Solution() and take no parameters.
    # Assume {dataset_name} and matplotlib.pyplot as plt are already available in the environment, so don't import any libraries.
    # The 'Solution()' function should create a {plot_type} of the {columns_used} columns,
    # include x and y labels, a title, and display the plot with plt.show().
    # Write the function ONCE, don't repeat it. Don't add any explanation after the function.

    Your function:\n
    """
    stats_prompt = f"""
    # Generate Python code only, DO NOT include any explanations or comments.
    # You have a DataFrame called {dataset_name} with columns: {df_columns}
    # The function should be named Stats() and take no parameters.
    # Assume {dataset_name} is already available in the environment.
    # Use {dataset_name}[{columns_used}].describe() to get the statistics information of the {columns_used} columns.
    # The 'Stats()' function should return the statistics information of the {columns_used} columns.
    # Write the function ONCE, don't repeat it. Don't add any explanation after the function.

    Your function starts below:\n
    """
    plot_input_tokens = tokenizer(code_prompt, return_tensors="pt", truncation=True, max_length=4096)
    plot_input_tokens = {k: v.to(model.device) for k, v in plot_input_tokens.items()}
    stats_input_tokens = tokenizer(stats_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    stats_input_tokens = {k: v.to(model.device) for k, v in stats_input_tokens.items()}
    return plot_input_tokens, stats_input_tokens

def tokenize_summary_prompt(plot_code, dataset_name, dataset, user_request, plot_type, stats):
    df_columns = list(dataset.columns)
    columns_used = re.findall(r"[\"'](.*?)[\"']", user_request)
    prompt = f"""
    # A table called {dataset_name} has these columns: {df_columns}
    # A {plot_type} involving the {columns_used} columns can be created with this python Function: {plot_code}
    # Here's the statistical information of the columns involved in this plot: {stats}
    # Now write a brief summary paragraph of what the {plot_type} looks like.

    Your summary:\n
    """
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_tokens = {k: v.to(model.device) for k, v in input_tokens.items()}
    return input_tokens

def generate_plot_code(tokenizer, model, plot_input_tokens):
    plot_input_len = plot_input_tokens["input_ids"].shape[1]
    plot_model_output = model.generate(**plot_input_tokens, max_new_tokens=100)
    plot_code = tokenizer.decode(plot_model_output[0, plot_input_len:], skip_special_tokens=True)
    plot_code = plot_code.replace("```python```", "").replace("```", "").strip()
    plot_code = plot_code.split("plt.show()")[0] + "plt.show()"
    return plot_code

def generate_stats_code(tokenizer, model, stats_input_tokens):
    stats_input_len = stats_input_tokens["input_ids"].shape[1]
    stats_model_output = model.generate(**stats_input_tokens, max_new_tokens=100)
    stats_code = tokenizer.decode(stats_model_output[0, stats_input_len:], skip_special_tokens=True)
    stats_code = stats_code.strip()
    stats_code = stats_code.split("describe()")[0] + "describe()"
    return stats_code

def generate_summary(summary_tokens):
    summary_input_len = summary_tokens["input_ids"].shape[1]
    model_output = model.generate(**summary_tokens, max_new_tokens=300)
    summary_texts = tokenizer.decode(model_output[0, summary_input_len:], skip_special_tokens=True)
    summary_texts = summary_texts.strip()
    return summary_texts

def execute_code(code, dataset_name, df):
    local_vars = {}
    global_vars = {dataset_name: df, "plt": plt, "pd": pd, "np": np, "sns": sns}
    try:
        exec(code, global_vars, local_vars)
        if "Solution" in local_vars:
            local_vars["Solution"]()
            return 1
        elif "Stats" in local_vars:
            stats = local_vars["Stats"]()
            return 1, stats
        else:
            print ("Generated function isn't named as 'Solution' or 'Stats'")
            return 0
    except Exception as e:
        print (f"Error executing the code: {e}")
        return 0

model_name = "Qwen/Qwen3-4B"
model, tokenizer = load_model_tokenizer(model_name)

def main():
    ev_data = "Data/electric_vehicles_spec_2025.csv"
    flower_data = "Data/Iris.csv"
    ev_df = load_datasets(ev_data)
    flower_df = load_datasets(flower_data)

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
    user_request = input("\nWhat do you want to explore about the dataset? Any columns mentioned should be enclosed in quotations. Enter your request:\n")
    plot_type = input("\nEnter the type of plot you want:\n")
    print ("\nThanks, your request is received. I'm currently generating code to complete your request.")

    plot_input_tokens, stats_input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request, plot_type)
    plot_code = generate_plot_code(tokenizer, model, plot_input_tokens)
    stats_code = generate_stats_code(tokenizer, model, stats_input_tokens)
    
    print (f"\nPlot code generated to complete your request: \n\t{plot_code}\n")
    plot_status = execute_code(plot_code, dataset_name, dataset)
    if plot_status == 1:
        print ("Your requested plots are successfully generated.\n")
    else:
        print ("Your requested plots failed to generate due to above errors.\n")
    
    stats_status, stats = execute_code(stats_code, dataset_name, dataset)
    if stats_status == 1:
        print (f"Here's the statistics information of the features included in your plot:\n{stats}")
        summary_tokens = tokenize_summary_prompt(plot_code, dataset_name, dataset, user_request, plot_type, stats)
        summary_texts = generate_summary(summary_tokens)
        print (f"Here's a brief summary paragraph about the plot made:\n{summary_texts}")
    else:
        print ("The statistics information failed to compute due to above errors.")

if __name__ == "__main__":
    main()
