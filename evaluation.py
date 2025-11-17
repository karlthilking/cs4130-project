### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

from project import load_datasets, load_model_tokenizer, tokenize_input, generate_code, execute_code
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

ev_df = load_datasets("UrvishAhir1/Electric-Vehicle-Specs-Dataset-2025")
flower_df = load_datasets("brjapon/iris")
model, tokenizer = load_model_tokenizer("Qwen/Qwen3-8B")

test_cases = [
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a bar chart showing the average 'top_speed_kmh' for each 'brand'."
    },
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a bar chart showing the average 'battery_capacity_kwh' for each 'battery_type'."
    },
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a scatter plot showing the relationship between 'top_speed_kmh' and 'acceleration_0_100_s'."
    },
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a scatter plot showing the relationship between 'efficiency_wh_per_km' and 'range_km'."
    },
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a density plot with two curves showing the distribution of 'length_mm' and 'Width_mm'."
    },
    {
        "dataset_name":  "ev_df",
        "dataset": ev_df,
        "user_request": "Give me a histogram showing the distribution of 'number_of_cells'."
    },
    {
        "dataset_name":  "flower_df",
        "dataset": flower_df,
        "user_request": "Give me a histogram showing the distribution of 'PetalWidthCm'."
    },
    {
        "dataset_name":  "flower_df",
        "dataset": flower_df,
        "user_request": "Give me a pie chart showing the distribution of 'Species'."
    },
    {
        "dataset_name":  "flower_df",
        "dataset": flower_df,
        "user_request": "Give me a frequency heat map of 'SepalLengthCm' VS 'SepalWidthCm'."
    },
    {
        "dataset_name":  "flower_df",
        "dataset": flower_df,
        "user_request": "Give me a frequency heat map of 'PetalLengthCm' VS 'PetalWidthCm'."
    }
]

def extract_function_name(code):
    try:
        parsed_code = ast.parse(code)
        function_names = [node.name for node in parsed_code.body if isinstance(node, ast.FunctionDef)]
        return function_names
    except Exception:
        return []

def has_no_parameter(code):
    try:
        parsed_code = ast.parse(code)
        nodes = [node for node in parsed_code.body if isinstance(node, ast.FunctionDef)]
        if len(nodes) == 1:
            return (len(nodes[0].args.args) == 0)
        else:
            return False
    except Exception:
        return False

def has_no_extra_code(code):
    try:
        parsed_code = ast.parse(code)
        nodes = [node for node in parsed_code.body]
        return (len(nodes) == 1 and isinstance(nodes[0], ast.FunctionDef))
    except Exception:
        return False
    
def correct_df_name(code):
    pass

def correct_columns(code):
    pass

def evaluate_code(code, dataset_name, dataset):
    original_show = plt.show
    plt.show = lambda: None
    result = execute_code(code, dataset_name, dataset)
    plt.show = original_show

    execution_success = (result == 1)
    func_names = extract_function_name(code)
    func_num_correct = (len(func_names) == 1)
    func_name_correct = (func_num_correct and func_names[0] == "Solution")
    no_parameter = has_no_parameter(code)
    no_extra_code = has_no_extra_code(code)
    df_name_correct = correct_df_name(code)
    col_names_correct = correct_columns(code)

    evaluation = {
        "Execution successful": execution_success,
        "Correct number of functions": func_num_correct,
        "Correct function name": func_name_correct,
        "Correct number of function parameters": no_parameter,
        "No extra reasoning/explanation/comments": no_extra_code,
        "Correct DataFrame name used": df_name_correct,
        "Correct column names used": col_names_correct
    }
    return evaluation

def main():
    for idx, test in enumerate(test_cases):
        dataset_name = test["dataset_name"]
        dataset = test["dataset"]
        user_request = test["user_request"]
        input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request)
        output_code = generate_code(tokenizer, model, input_tokens)
        evaluation = evaluate_code(output_code, dataset_name, dataset)
        print (f"Evaluation of agent output code from test case {idx}:\n{evaluation}\n\n")

if __name__ == "__main__":
    main()
