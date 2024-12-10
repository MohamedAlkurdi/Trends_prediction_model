import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
labels = ["bussiness","technology","science","sports","media and entertainment","politics","health","crime","accident","environment","art","literature","tragedy","education","fashion","food","travel","military","real estates","history","religion","celebrity"]
hypothesis_template = "this text is about {}"

# data = pd.read_csv('../preprocessed_data/cleaned_data_USA.csv')
# data = pd.read_csv('../preprocessed_data/cleaned_data_Nigeria.csv')
data = pd.read_csv('../preprocessed_data/cleaned_data_Malaysia.csv')
dataFrame = data[['date','traffic','newsSnippet']]
strings_list = [item[2] for item in dataFrame.values.tolist()]
loop = range(len(strings_list))

results = []
batch_size = 4
for i in range(0, len(strings_list), batch_size):
    batch = strings_list[i:i+batch_size]
    predictions = classifier(batch, labels, hypothesis_template=hypothesis_template, multi_class=True)
    for j, prediction in enumerate(predictions):
        top_label = prediction['labels'][0]
        top_score = prediction['scores'][0]
        row_data = (dataFrame.iloc[i+j]['date'], dataFrame.iloc[i+j]['traffic'], dataFrame.iloc[i+j]['newsSnippet'], top_label, top_score)
        results.append(row_data)

df_results = pd.DataFrame(results, columns=['date', 'traffic', 'newsTitle', 'predicted_label', 'score'])
df_results.to_csv('./classification_output.csv', index=False)

def clean_traffic(value):
    value = value.replace(",", "").replace("+", "").strip()
    return int(value)

data = pd.read_csv('./Malaysia_classification_output.csv')
df = data[['predicted_label','traffic']]
df['traffic_numeric'] = df['traffic'].apply(clean_traffic)

topic_stats = df.groupby('predicted_label')['traffic_numeric'].agg(['mean']).sort_values('mean',ascending=True)
topic_stats.plot(kind='bar', color='skyblue')
