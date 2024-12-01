import pandas as pd
import matplotlib.pyplot as plt

# entity_counts_over_time = pd.read_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\NER\NER_statisctis_tables\entity_counts_every_day.csv')
entity_counts_over_time = pd.read_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\NER\NER_statisctis_tables\entity_counts_every_day_TEST.csv')

# entity_counts_over_time[['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']].plot(kind='line', figsize=(12, 6))
entity_counts_over_time[['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LAW', 'LOC', 'NORP', 'ORDINAL', 'ORG', 'PERSON', 'PRODUCT', 'TIME', 'WORK_OF_ART']].plot(kind='line', figsize=(12, 6))


plt.title('Entity Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.legend(title='Entity Type')
plt.show()
