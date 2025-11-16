### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_datasets(dataset):
    data = load_dataset(dataset, split="train")
    df = data.to_pandas()
    return df

def load_model_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_input(tokenizer, model, dataset_name: str, dataset, user_request):
    df_columns = list(dataset.columns)
    condition_1 = f"You have a DataFrame called {dataset_name} with these columns: {df_columns}. Using this dataset, you will be asked to generate some plots for \
exploration purpose so the user can better understand the dataset. \nUser requirement: "
    condition_2 = "\nBased on the requirement, generate a python function called Solution() without any parameters that generate the requested plots. When generating \
plots, use the Matplotlib library and the function shouldn't return anything but simply show the plot direcrly using plt.show(). Everything besides the fucntion itself \
should not be included, so any comments, explanation, and reasoning are not needed."
    condition_3 = "\nAfter the plot is shown, use the .describe() function in pandas to get the key statistics information about the DataFrame. You should print out the \
statistics of the features/columns used in the plot, and print a brief summary paragraph about the plot. For example, given a health dataframe called health_df and the \
user request of 'Give me a scatter plot to show the relationship between height and weight', you should first generate and show the plot, then you find the statistics \
of the height and weight column using health_df.describe(), and finally write a short paragraph summarizing the plot. \n\nYour function:\n"
    prompt = condition_1 + user_request + condition_2 + condition_3
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    return input_tokens

def generate_code(tokenizer, model, input_tokens):
    input_len = len(input_tokens.input_ids[0])
    model_output = model.generate(**input_tokens, max_new_tokens=300)
    output_code = tokenizer.decode(model_output[input_len:], skip_special_tokens=True)
    output_code = output_code.replace("```python```", "").replace("```", "")
    output_code = output_code.strip()
    return output_code

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
    user_request = input("\nEnter your request:\n")

    input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request)
    output_code = generate_code(tokenizer, model, input_tokens)
    
    print ("\nThanks, your request is received. I'm currently generating code to complete your request.")
    print (f"\n\nCode generated to complete your request: \n{output_code}\n\n")
    result = execute_code(output_code, dataset_name, dataset)
    if result == 1:
        print ("Your requested plots are successfully generated.")
    else:
        print ("Your requested plots failed to generate due to some errors.")
    
if __name__ == "__main__":
    main()
