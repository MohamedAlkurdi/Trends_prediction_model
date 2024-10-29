import spacy
import pandas as pd

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


#------------------------------------------------
df_entities = pd.DataFrame(entities_data)

df_entities['date'] = pd.to_datetime(df_entities['date'])

exploded_df = df_entities.explode('entities')

entity_counts_every_day = exploded_df.groupby(['date', 'entities']).size().unstack(fill_value=0)

entity_counts_every_day.to_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\entity_counts_every_day.csv', header=True, index=True)


#------------------------------------------------

df_entities_traffic = pd.DataFrame(entities_data)

df_entities_traffic['traffic'] = pd.to_numeric(df_entities_traffic['traffic'])



