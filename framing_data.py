import pandas as pd
import re
import json
import os

jsonPath = './dataset/countries.json'
csvPath = "C:/Users/alkrd/Desktop/graduation_project/the_project/dataset/primearchive.blogspot.com_detailled-trends_all-countries.csv"
output_dir = "C:/Users/alkrd/Desktop/graduation_project/the_project/preprocessed_data"

with open(jsonPath, 'r') as file:
    data = json.load(file)
countries = data["countries"]

def clean_text(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.strip()  
    return text

df = pd.read_csv(csvPath)

os.makedirs(output_dir, exist_ok=True)

for country in countries:
    filtered_df = df[df['country'] == country]  # Filter by country
    if not filtered_df.empty:  # Proceed only if there is data for the country
        filtered_df.loc[:, 'name'] = filtered_df['name'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'relatedKeyword'] = filtered_df['relatedKeyword'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'newsTitle'] = filtered_df['newsTitle'].astype(str).apply(clean_text)
        filtered_df.loc[:, 'newsSnippet'] = filtered_df['newsSnippet'].astype(str).apply(clean_text)

        cleaned_df = filtered_df[['index', 'dayId', 'date', 'name', 'traffic', 'newsTitle', 'newsSnippet']]

        output_path = os.path.join(output_dir, f"cleaned_data_{country}.csv")
        cleaned_df.to_csv(output_path, index=False)
