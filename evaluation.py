### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

from project import load_datasets, load_model_tokenizer, tokenize_input, tokenize_summary_prompt, generate_plot_code, generate_stats_code, generate_summary, execute_code
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.image import AxesImage
from matplotlib.patches import Wedge
from test_case import test_cases
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import re

ev_df = load_datasets("Data/electric_vehicles_spec_2025.csv")
flower_df = load_datasets("Data/Iris.csv")
model, tokenizer = load_model_tokenizer("Qwen/Qwen3-4B")
dataset_name_map = {"ev_df": ev_df, "flower_df": flower_df}
for test in test_cases:
    test["dataset"] = dataset_name_map[test["dataset_name"]]

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
    
def correct_df_name(code, dataset_name):
    try:
        parsed_code = ast.parse(code)
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id != dataset_name:
                    return False
        return True
    except Exception:
        return False

def correct_columns(code, dataset):
    try:
        parsed_code = ast.parse(code)
        df_columns = list(dataset.columns)
        column_used = []
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Constant) and isinstance(node.s, str):
                column_used.append(node.s)
        for column in column_used:
            if column not in df_columns:
                return False
        return True
    except Exception:
        return False
    
def correct_title_labels(code, dataset_name, dataset):
    original_show = plt.show
    plt.show = lambda: None
    local_vars = {}
    global_vars = {dataset_name: dataset, "plt": plt, "pd": pd, "np": np}
    try:
        exec(code, global_vars, local_vars)
        if "Solution" in local_vars:
            local_vars["Solution"]()
            ax = plt.gca()
            has_title = bool(ax.get_title())
            has_xlabel = bool(ax.get_xlabel())
            has_ylabel = bool(ax.get_ylabel())
            plt.show = original_show
            if has_title and has_xlabel and has_ylabel:
                return True
            return False
        else:
            plt.show = original_show
            return False
    except Exception:
        plt.show = original_show
        return False

def find_plot_type(code, dataset_name, dataset):
    original_show = plt.show
    plt.show = lambda: None
    local_vars = {}
    global_vars = {dataset_name: dataset, "plt": plt, "pd": pd, "np": np}
    try:
        exec(code, global_vars, local_vars)
        if "Solution" in local_vars:
            local_vars["Solution"]()
            ax = plt.gca()
            if len(ax.lines) > 0:
                return "line plot"
            elif ax.collections and isinstance(ax.collections[0], PathCollection):
                return "scatter plot"
            elif (ax.images and isinstance(ax.images[0], AxesImage)) or (ax.collections and isinstance(ax.collections[0], QuadMesh)):
                return "heat map"
            elif ax.patches and all(isinstance(p, Wedge) for p in ax.patches):
                return "pie chart"
            elif ax.patches:
                widths = {round(p.get_width(), 5) for p in ax.patches}
                if len(widths) == 1 and list(widths)[0] < 0.5:
                    return "histogram"
                return "bar chart"
            return "unknown"
        else:
            return "Unknown"
    except Exception:
        plt.show = original_show
        return "Unknown"
    finally:
        plt.show = original_show

def correct_statistics(stats, user_request, dataset):
    stats_block = stats.split("Summary")[0].strip()
    columns_used = re.findall(r"[\"'](.*?)[\"']", user_request)
    existing_columns_used = [col for col in columns_used if col in list(dataset.columns)]
    correct_stats = dataset[existing_columns_used].describe().to_dict()
    try:
        parsed_stats = ast.literal_eval(stats_block)
    except:
        return False
    for column in existing_columns_used:
        if column not in parsed_stats:
            return False
        for metric, value in correct_stats[column].items():
            if metric not in parsed_stats[column]:
                return False
            if not np.isclose(float(parsed_stats[column][metric]), float(value), atol=1e-6, rtol=1e-6):
                return False
    return True

def evaluate_code(code, stats, dataset_name, dataset, plot_type, user_request):
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
    df_name_correct = correct_df_name(code, dataset_name)
    col_names_correct = correct_columns(code, dataset)
    title_and_labels_correct = correct_title_labels(code, dataset_name, dataset)
    plot_type_found = find_plot_type(code, dataset_name, dataset)
    if plot_type_found == "line plot":
        plot_type_correct = True if plot_type in ["line plot", "density plot"] else False
    else:
        plot_type_correct = (plot_type == find_plot_type(code, dataset_name, dataset))
    statistics_correct = correct_statistics(stats, user_request, dataset)

    evaluation = {
        "Execution successful": execution_success,
        "Correct number of functions": func_num_correct,
        "Correct function name": func_name_correct,
        "Correct number of function parameters": no_parameter,
        "No extra reasoning/explanation/comments": no_extra_code,
        "Correct plot type": plot_type_correct,
        "Correct DataFrame name used": df_name_correct,
        "Correct column names used": col_names_correct,
        "Titles and x, y labels properly created": title_and_labels_correct,
        "Correct dataset statistics": statistics_correct
    }
    return evaluation

def main():
    for idx, test in enumerate(test_cases):
        dataset_name = test["dataset_name"]
        dataset = test["dataset"]
        user_request = test["user_request"].lower()
        plot_type = test["plot_type"].lower()
        plot_input_tokens, stats_input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request, plot_type)
        plot_code = generate_plot_code(tokenizer, model, plot_input_tokens)
        stats_code = generate_stats_code(tokenizer, model, stats_input_tokens)
        stats = execute_code(stats_code, dataset_name, dataset)[1]
        evaluation = evaluate_code(plot_code, stats, dataset_name, dataset, plot_type, user_request)
        print (f"Evaluation of agent output code from test case {idx}:\n{evaluation}\n\n")

if __name__ == "__main__":
    main()
