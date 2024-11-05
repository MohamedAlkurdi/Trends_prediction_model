import pandas as pd
import matplotlib.pyplot as plt

entity_counts_over_time = pd.read_csv(r'C:\Users\alkrd\Desktop\graduation_project\the_project\traffic_entities_ratio.csv', index_col=0)

entity_counts_over_time_weighted = entity_counts_over_time.multiply(entity_counts_over_time.index, axis=0)

total_traffic_per_entity = entity_counts_over_time_weighted.sum(axis=0)

total_traffic_per_entity.plot(kind='bar', figsize=(12, 6), color='skyblue')

plt.title('Total Traffic by Entity Type')
plt.xlabel('Entity Type')
plt.ylabel('Total Traffic Weighted by Occurrence')
plt.xticks(rotation=45)
plt.show()
