### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/12/2025

from project import load_datasets, load_model_tokenizer, tokenize_input, tokenize_summary_prompt, generate_plot_code, generate_stats_code, generate_summary, execute_code
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.image import AxesImage
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
from tests import test_cases
import seaborn as sns
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
            if isinstance(node, ast.Name) and node.id == dataset_name:
                    return True
        return False
    except Exception:
        return False

def correct_columns(code, dataset):
    try:
        parsed_code = ast.parse(code)
        df_columns = list(dataset.columns)
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name):
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        if node.slice.value not in df_columns:
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
    if "plt.plot(" in code:
        return "line plot"
    elif "plt.bar(" in code:
        return "bar chart"
    elif "plt.hist(" in code:
        return "histogram"
    elif "plt.scatter(" in code:
        return "scatter plot"
    elif "plt.boxplot(" in code:
        return "boxplot"
    elif "plt.violinplot(" in code:
        return "violin plot"
    elif "plt.pie(" in code:
        return "bar chart"
    elif "sns.kdeplot(" in code:
        return "density plot"
    else:
        return "Unknown"

def correct_statistics(stats, user_request, dataset):
    parsed_stats = stats.to_dict()
    columns_used = re.findall(r"[\"'](.*?)[\"']", user_request)
    existing_columns_used = [col for col in columns_used if col in list(dataset.columns)]
    if not existing_columns_used:
        return False
    correct_stats = dataset[existing_columns_used].describe().to_dict()
    for column in existing_columns_used:
        if column not in parsed_stats:
            return False
        for metric, value in correct_stats[column].items():
            if metric not in parsed_stats[column]:
                return False
            try:
                a = float(parsed_stats[column][metric])
                b = float(value)
                if not np.isclose(a, b, atol=1e-6, rtol=1e-6):
                    return False
            except (TypeError, ValueError):
                if str(parsed_stats[column][metric]) != str(value):
                    return False
    return True

def compute_pass_rate(evaluations):
    pass_rate_dict = {}
    metrics = ["Execution successful", "Correct number of functions", "Correct function name", "Correct number of function parameters",
               "No extra reasoning/explanation/comments", "Correct plot type", "Correct DataFrame name used", "Correct column names used",
               "Titles and x, y labels properly created", "Correct dataset statistics"]
    for metric in metrics:
        pass_rate_dict[metric] = 0
    for evaluation in evaluations:
        for key, value in evaluation.items():
            pass_rate_dict[key] += 1 if value else 0
    for metric in metrics:
        pass_rate_dict[metric] = round(pass_rate_dict[metric] / len(evaluations), 3)
    return pass_rate_dict

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
    statistics_correct = correct_statistics(stats, user_request, dataset)

    evaluation = {
        "Execution successful": execution_success,
        "Correct number of functions": func_num_correct,
        "Correct function name": func_name_correct,
        "Correct number of function parameters": no_parameter,
        "No extra reasoning/explanation/comments": no_extra_code,
        "Correct DataFrame name used": df_name_correct,
        "Correct column names used": col_names_correct,
        "Titles and x, y labels properly created": title_and_labels_correct,
        "Correct dataset statistics": statistics_correct
    }
    return evaluation

def main():
    all_evaluations = []
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
        all_evaluations.append(evaluation)
        print (f"Evaluation of agent output code from test case {idx}:\n{evaluation}\n")
    pass_rate = compute_pass_rate(all_evaluations)
    print (f"Pass rate of all 50 test cases for each evaluation metrics:\n{pass_rate}")

if __name__ == "__main__":
    main()
