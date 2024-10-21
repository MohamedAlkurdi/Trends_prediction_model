import pandas as pd
import re

path = r"C:\Users\alkrd\Desktop\graduation_project\the_project\dataset\primearchive.blogspot.com_detailled-trends_all-countries.csv"
cleaned_data = r"C:\Users\alkrd\Desktop\graduation_project\the_project\preprocssed_data\cleaned_data_USA.csv"
country = "USA"

df = pd.read_csv(path)
filtered_df = df[df['country'] == country]

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

filtered_df['name'] = filtered_df['name'].apply(clean_text)
# filtered_df['relatedKeyword'] = filtered_df['relatedKeyword'].apply(clean_text)
filtered_df['newsTitle'] = filtered_df['newsTitle'].apply(clean_text)
filtered_df['newsSnippet'] = filtered_df['newsSnippet'].apply(clean_text)

cleaned_df = filtered_df[['index', 'dayId', 'date', 'name', 'traffic', 'newsTitle', 'newsSnippet']]

cleaned_df.to_csv(cleaned_data, index=False)
