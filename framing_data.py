import pandas as pd
import re
import json
import os

# Path to the JSON file and CSV file
jsonPath = r'./dataset/countries.json'
csvPath = r"C:\Users\alkrd\Desktop\graduation_project\the_project\dataset\primearchive.blogspot.com_detailled-trends_all-countries.csv"
output_dir = r"C:\Users\alkrd\Desktop\graduation_project\the_project\preprocessed_data"

# Load countries data
with open(jsonPath, 'r') as file:
    data = json.load(file)
countries = data["countries"]

# Define a function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Strip leading/trailing whitespace
    return text

# Load the dataset
df = pd.read_csv(csvPath)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through countries and clean/filter data
for country in countries:
    filtered_df = df[df['country'] == country]  # Filter by country
    if not filtered_df.empty:  # Proceed only if there is data for the country
        filtered_df.loc[:, 'name'] = filtered_df['name'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'relatedKeyword'] = filtered_df['relatedKeyword'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'newsTitle'] = filtered_df['newsTitle'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'newsSnippet'] = filtered_df['newsSnippet'].astype(str).apply(clean_text)

        # Select desired columns
        cleaned_df = filtered_df[['index', 'dayId', 'date', 'name', 'traffic', 'newsTitle', 'newsSnippet']]

        # Save to a CSV file
        output_path = os.path.join(output_dir, f"cleaned_data_{country}.csv")
        cleaned_df.to_csv(output_path, index=False)
