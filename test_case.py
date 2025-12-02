test_cases = [
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a bar chart showing the average 'top_speed_kmh' for each 'brand'.",
        "plot_type": "Bar chart"
    },
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a bar chart showing the average 'battery_capacity_kwh' for each 'battery_type'.",
        "plot_type": "Bar chart"
    },
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a scatter plot showing the relationship between 'top_speed_kmh' and 'acceleration_0_100_s'.",
        "plot_type": "Scatter plot"
    },
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a scatter plot showing the relationship between 'efficiency_wh_per_km' and 'range_km'.",
        "plot_type": "Scatter plot"
    },
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a density plot with two curves showing the distribution of 'length_mm' and 'Width_mm'.",
        "plot_type": "Density plot"
    },
    {
        "dataset_name":  "ev_df",
        "user_request": "Give me a histogram showing the distribution of 'number_of_cells'.",
        "plot_type": "Histogram"
    },
    {
        "dataset_name":  "flower_df",
        "user_request": "Give me a histogram showing the distribution of 'PetalWidthCm'.",
        "plot_type": "Histogram"
    },
    {
        "dataset_name":  "flower_df",
        "user_request": "Give me a pie chart showing the distribution of 'Species'.",
        "plot_type": "Pie chart"
    },
    {
        "dataset_name":  "flower_df",
        "user_request": "Give me a frequency heat map of 'SepalLengthCm' VS 'SepalWidthCm'.",
        "plot_type": "Heat map"
    },
    {
        "dataset_name":  "flower_df",
        "user_request": "Give me a frequency heat map of 'PetalLengthCm' VS 'PetalWidthCm'.",
        "plot_type": "Heat map"
    }
]
