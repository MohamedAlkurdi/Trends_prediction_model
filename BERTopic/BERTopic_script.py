from bertopic import BERTopic
import pandas as pd
data = pd.read_csv("../preprocssed_data/cleaned_data_USA.csv")    
# dataFrame = data[['newsSnippet']]
dataFrame = data[['newsTitle']]
strings_list = [item[0] for item in dataFrame.values.tolist()]
topic_model = BERTopic(embedding_model='all-MiniLM-L6-v2')
topics,probs = topic_model.fit_transform(strings_list)
topic_model.get_topic_info() #40 topıc clusters
df = pd.DataFrame({'topic':topics, 'documents':probs})
topic_model.visualize_topics()