### Kuan-Chun Chiu, Karl Thilking
### Prof. Guha
### CS 4130
### 12/05/2025

from project import load_datasets, load_model_tokenizer, tokenize_input, generate_code, execute_code

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

def evaluate_code(code):
    pass

def main():
    for idx, test in enumerate(test_cases):
        dataset_name = test["dataset_name"]
        dataset = test["dataset"]
        user_request = test["user_request"]
        input_tokens = tokenize_input(tokenizer, model, dataset_name, dataset, user_request)
        output_code = generate_code(tokenizer, model, input_tokens)
        result = evaluate_code(output_code)
        print (f"Evaluation of agent output code from test case {idx}:\n{result}\n\n")

if __name__ == "__main__":
    main()
