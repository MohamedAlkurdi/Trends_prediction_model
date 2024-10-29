import spacy
import pandas as pd

# entity_counts_every_day_output = r"C:\Users\alkrd\Desktop\graduation_project\the_project\entity_counts_every_day.txt"

nlp = spacy.load("en_core_web_sm")
data = pd.read_csv(r"C:\Users\alkrd\Desktop\graduation_project\the_project\preprocssed_data\cleaned_data_USA.csv")

dataFrame = data[['index', 'date', 'traffic', 'newsTitle', 'newsSnippet']]

entities_data = []

for _, row in dataFrame.iterrows():
    doc = nlp(row['newsTitle'] + " " + row['newsSnippet'])
    
    entities = [ent.label_ for ent in doc.ents]
    
    entities_data.append({
        'id': row['index'],
        'date': row['date'],
        'traffic': row['traffic'],
        'entities': entities 
    })

df_entities = pd.DataFrame(entities_data)

df_entities['date'] = pd.to_datetime(df_entities['date'])

# Expand the 'entities' list into individual rows for counting
exploded_df = df_entities.explode('entities')

# Group by date and entity type, then count occurrences
entity_counts_every_day = exploded_df.groupby(['date', 'entities']).size().unstack(fill_value=0)

# Display the result to show entity frequencies over time

print(entity_counts_every_day)


entity_counts_every_day.to_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\entity_counts_every_day.csv', header=True, index=True)
entity_counts_every_day.to_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\entity_counts_every_day.txt', header=True, index=True)



