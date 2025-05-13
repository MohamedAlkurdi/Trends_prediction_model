import streamlit as st
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

original_dataset = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/dataset/primearchive.blogspot.com_detailled-trends_all-countries.csv")
original_dataset_split_output_example = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/preprocessed_data/cleaned_data_Belgium.csv")
ner_statisctis1 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/NER/NER_statisctis_tables/entity_counts_every_day.csv")
ner_statisctis2 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/NER/NER_statisctis_tables/traffic_entities_ratio.csv")

initial_classification_output_example1 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/classified_data/Australia_classification_output.csv")
initial_classification_output_example2 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/europe/classified_data/Denmark_classification_output.csv")
initial_classification_output_example3 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/africa/classified_data/Kenya_classification_output.csv")


clustered_classification_output_example1 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/clustered_classified_data/Canada_clustered_classified.csv")
clustered_classification_output_example2 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/europe/clustered_classified_data/Finland_clustered_classified.csv")
clustered_classification_output_example3 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/africa/clustered_classified_data/Nigeria_clustered_classified.csv")

clustered_classified_data_with_relative_traffic_rates1 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/africa/genral_labeled_data_with_relative_traffic_rates/Economy/real_data/SouthAfrica_with_relative_traffic_rates.csv")
clustered_classified_data_with_relative_traffic_rates2 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/europe/genral_labeled_data_with_relative_traffic_rates/Geopolitical/real_data/UK_with_relative_traffic_rates.csv")
clustered_classified_data_with_relative_traffic_rates3 = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/real_data/usa_with_relative_traffic_rates.csv")

regressor_example = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/immigration.csv")

google_trend_data_sample = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/google_trends/exmaple_for_streamlit.csv")

initial_prophet_model_results = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/initial_training_results.csv")

# from Classification.similarity_values_calculated_with_cosine import cosine_similarity_values

cosine_similarity_values = {
    "AFRICA": {
        "Economy": pd.DataFrame(
            {
                "Kenya": ["-", 99.37, 98.1],
                "Nigeria": [99.37, "-", 97.47],
                "SouthAfrica": [98.1, 97.47, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Kenya": ["-", 80.38, 81.01],
                "Nigeria": [80.38, "-", 79.75],
                "SouthAfrica": [81.01, 79.75, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Kenya": ["-", 60.13, 60.76],
                "Nigeria": [60.13, "-", 59.49],
                "SouthAfrica": [60.76, 59.49, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Kenya": ["-", 70.89, 58.23],
                "Nigeria": [70.89, "-", 57.59],
                "SouthAfrica": [58.23, 57.59, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Accident": pd.DataFrame(
            {
                "Kenya": ["-", 55.7, 48.1],
                "Nigeria": [55.7, "-", 49.37],
                "SouthAfrica": [48.1, 49.37, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Kenya": ["-", 72.78, 63.92],
                "Nigeria": [72.78, "-", 62.03],
                "SouthAfrica": [63.92, 62.03, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Kenya": ["-", 65.82, 57.59],
                "Nigeria": [65.82, "-", 57.59],
                "SouthAfrica": [57.59, 57.59, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
    },
    "EUROPE": {
        "Economy": pd.DataFrame(
            {
                "Denmark": ["-", 92.41, 88.61],
                "UK": [92.41, "-", 93.67],
                "Finland": [88.61, 93.67, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Denmark": ["-", 53.8, 56.96],
                "UK": [53.8, "-", 50.63],
                "Finland": [56.96, 50.63, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Denmark": ["-", 77.85, 86.71],
                "UK": [77.85, "-", 73.42],
                "Finland": [86.71, 73.42, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Denmark": ["-", 50.0, 48.1],
                "UK": [50.0, "-", 46.2],
                "Finland": [48.1, 46.2, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Accident": pd.DataFrame(
            {
                "Denmark": ["-", 51.9, 49.37],
                "UK": [51.9, "-", 60.76],
                "Finland": [49.37, 60.76, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Denmark": ["-", 48.73, 44.94],
                "UK": [48.73, "-", 43.67],
                "Finland": [44.94, 43.67, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Denmark": ["-", 50.0, 44.3],
                "UK": [50.0, "-", 46.2],
                "Finland": [44.3, 46.2, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
    },
    "NORTH_AMERICA_AUSTRALIA": {
        "Economy": pd.DataFrame(
            {
                "Australia": ["-", 94.94, 96.84],
                "Canada": [94.94, "-", 95.57],
                "USA": [96.84, 95.57, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Australia": ["-", 50.63, 55.7],
                "Canada": [50.63, "-", 59.49],
                "USA": [55.7, 59.49, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Australia": ["-", 91.14, 85.44],
                "Canada": [91.14, "-", 91.77],
                "USA": [85.44, 91.77, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Australia": ["-", 68.35, 72.15],
                "Canada": [68.35, "-", 65.19],
                "USA": [72.15, 65.19, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Accident": pd.DataFrame(
            {
                "Australia": ["-", 74.05, 79.75],
                "Canada": [74.05, "-", 74.68],
                "USA": [79.75, 74.68, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Australia": ["-", 46.84, 47.47],
                "Canada": [46.84, "-", 62.03],
                "USA": [47.47, 62.03, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Australia": ["-", 58.23, 70.25],
                "Canada": [58.23, "-", 55.7],
                "USA": [70.25, 55.7, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
    },
    "WEST_ASIA": {
        "Economy": pd.DataFrame(
            {
                "Australia": ["-", 95.57, 93.67],
                "Canada": [95.57, "-", 94.3],
                "USA": [93.67, 94.3, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Australia": ["-", 67.72, 60.76],
                "Canada": [67.72, "-", 63.29],
                "USA": [60.76, 63.29, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Australia": ["-", 78.48, 81.65],
                "Canada": [78.48, "-", 91.77],
                "USA": [81.65, 91.77, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Australia": ["-", 62.66, 52.53],
                "Canada": [62.66, "-", 44.3],
                "USA": [52.53, 44.3, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Accident": pd.DataFrame(
            {
                "Australia": ["-", 52.53, 49.37],
                "Canada": [52.53, "-", 47.47],
                "USA": [49.37, 47.47, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Australia": ["-", 76.58, 55.06],
                "Canada": [76.58, "-", 50.63],
                "USA": [55.06, 50.63, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Australia": ["-", 58.86, 66.46],
                "Canada": [58.86, "-", 55.06],
                "USA": [66.46, 55.06, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
    },
}

def space(height=1):
    value = ""
    for i in range(height):
        value += "<br/>"
    return st.html(value)


def main():

    st.set_page_config(layout="wide")
    st.title("🧮 Audience Interest Trend Forecasting")

    space(2)
    
    st.markdown("""
                **By**: [Mohammed Alkurdi](https://muhammedalkurdiportfolio.vercel.app/en)\n
                **Instrcutor**: [DR. MUSTAFA HAKAN BOZKURT](https://avesis.ktu.edu.tr/mhakanbozkurt)
                """)

    st.header("📍 Introduction")
    st.markdown(
        """
    The overall goal of this project is attempting to predict the future performance of communities based on geographic location and cultural interests, in order to support better decision-making processes targeting society as a whole.  

    Rather than predicting precise numerical performance, this project focuses on forecasting the fluctuations in public interest toward specific topics within given geographic regions.

    It is important to mention that I was not exactly trying to solve a specific real-world problem, so my work is an extended first step in the field of decision-making support processes using real-world data examples.

    Below, I will take you on a quick journey through the steps of implementing this project. During this journey, you will notice developments in the project's working method, which I hope reflect my ability to deal with unexpected changes.
    """
    )

    space(2)

    st.subheader("🎯 Project Objectives")
    st.markdown(
        """
    1. To solve a problem that can be applied in the real world while ensuring its scalability.
    2. To test the feasibility of predicting community behavior and exploiting this knowledge to tailor public-facing actions.
    3. To attempt to predict trends in interest related to a specific topic in a specific geographic area.
    """
    )

    space(2)

    st.header("📍 CHAPTER 1: Early Data Preprocessing")

    st.subheader("1.1 - Original dataset")
    st.markdown(
        "The dataset below was our starting point and contains data for news snippets collected from around the world, along with information about news interactions, their date, source, and more."
    )
    st.markdown(
        "Dataset: [Daily Global Trends 2020 on Kaggle](https://www.kaggle.com/datasets/thedevastator/daily-global-trends-2020-insights-on-popularity)"
    )
    st.write(original_dataset)
    
    space() 
    st.subheader("✒️ Credit")
    st.markdown("The author of the dataset that I started my project using it is [Jeffrey Mvutu Mabilama.](https://data.world/jfreex)")


    space()

    st.subheader("1.2 - Origin-based split")
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Original Dataset Splitting Loop:**
        ```python
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
        ```
        """
        )

    space()

    st.subheader("1.2.1 - Output example")
    st.write(original_dataset_split_output_example)

    space()

    st.subheader("1.3 - English only")
    st.markdown(
        """
    After splitting the countries, I decided to target only the English data since the Natural Language Processing tool supports English perfectly. It turned out that there are 12 countries that publish their news in English, and fortunately, each group of three belongs to regions that are either geographically or culturally close.
    """
    )

    st.dataframe(
        {
            "Region": [
                "Africa",
                "East Asia",
                "North America & Australia",
                "Europe",
            ],
            "Countries": [
                "Kenya, Nigeria, South Africa",
                "Malaysia, Philippines, Singapore",
                "USA, Canada, Australia",
                "UK, Denmark, Finland",
            ],
        }
    )

    space(2)

    st.header("📍 CHAPTER 2: Data Exploration")

    st.subheader("2.1 - Named Entity Recognition")
    st.markdown(
        "My first experience with data analysis was with the 'Named Entity Recognition' technique, which gave me a morale boost after seeing its results. This provided a good overview, but it became clear that this process was not sufficient."
    )
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Original Dataset Splitting Loop:**
        ```python
        nlp = spacy.load("en_core_web_sm")
        data = pd.read_csv("C:/Users/alkrd/Desktop/graduation_project/the_project/preprocssed_data/cleaned_data_USA.csv")

        dataFrame = data[['index', 'date','name','traffic', 'newsTitle', 'newsSnippet']] # added name columns

        entities_data = []

        for _, row in dataFrame.iterrows():
            doc = nlp(row['name'])
            
            entities = [ent.label_ for ent in doc.ents]
            
            entities_data.append({
                'id': row['index'],
                'date': row['date'],
                'traffic': row['traffic'],
                'entities': entities 
            })
        ```
        """
        )

    space()

    st.markdown(
        "In this experiment, I studied the relationship between the interaction rate and the 'named entities' on the one hand, and extracted the relationship between time and the reaction rate on the other hand."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(ner_statisctis1, height=400)  # Set a fixed height for the table
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/NER/visuals/visualize_entity_count_everyday.png"
        )

    with col2:
        st.dataframe(ner_statisctis2, height=400)  # Set the same height for consistency
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/NER/visuals/visualize_traffic_entities_ratio.png"
        )

    space()

    st.subheader("2.2 - Topic Modelling")
    st.markdown(
        "After noticing the limited results of the 'Named Entity Recognition' process, I decided to try another method that focused on analyzing natural language and extracting the most frequent topics, a process called 'topic modeling'."
    )
    st.markdown(
        "In this process, I tried several techniques, modifying the inputs and configuration variables many times in an attempt to achieve the best results."
    )
    st.markdown(
        "I first used the LDA algorithm and later used the BERTopic model for the same purpose."
    )

    st.markdown("### 2.2.1 - USING LDA")

    with st.expander("Code Snippet"):
        st.markdown(
            """
        **LDA Model Setup:**
        ```python
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=30,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=20,
            alpha="auto"
        )
        ```
        """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/Topic_Modeling/LDA_topic_modeling/visuals/lda_model_vis1.png"
        )

    with col2:
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/Topic_Modeling/LDA_topic_modeling/visuals/lda_model_vis_2.png"
        )

    space()

    st.markdown("### 2.2.2 - USING BERTopic")

    with st.expander("Code Snippet"):
        st.markdown(
            """
        **LDA Model Setup:**
        ```python
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(strings_list)
        topic_model.update_topics(strings_list, n_gram_range=(1,3))
        ```
        """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/Topic_Modeling/BERTopic/newplot.png",
            use_container_width=True,
        )

    with col2:
        st.image(
            "C:/Users/alkrd/Desktop/graduation_project/the_project/Topic_Modeling/BERTopic/newplot2.png",
            use_container_width=True,
        )

    space(2)

    st.header("📍 CHAPTER 3: Critical Redirection")

    st.subheader("3.1 - Zero-shot Classification")
    space()
    st.subheader("3.1.1 - Initial Classification")
    
    st.markdown(
        """
        So far, I was using methods that belong to the "Unsupervised Machine Learning" style, which depends on recognizing patterns and trying to extract an underlying meaning from data. Apparently, this is not the best option.
        So, the next step was to consider taking the "Supervised Learning" approach, which means I would label the data and later train the model on these labels so it could classify inputs after learning.
        To do the "labeling" task, there were a few options:
        1. Labeling the data manually, which can take a massive amount of time.
        2. Using a pretrained classification model that can classify your data, and indeed that is what happened.
        So, I started looking for a pretrained classification model and found one soon, allowing me to start the most important phase in my project.
        The technique used is called "zero-shot classification," and it is defined as "a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes".
        """
    )

    space()

    st.markdown(
        "Using Hugging Face zero-shot models: [Link](https://huggingface.co/tasks/zero-shot-classification)"
    )
    st.markdown("Classification Categories:")
    st.json(
        {
            "Categories": "bussiness ,real estates ,technology ,science ,media and entertainment ,art ,celebrity ,sports ,environment ,health ,fashion ,travel ,food ,tragedy ,crime ,accident ,politics ,military ,education ,literature ,history ,religion"
        }
    )
    st.markdown("How Classification Model Works:")
    st.json({"Sample": "Dune is the best movie ever.", "Labels": "CINEMA, ART, MUSIC", "Scores":"0.900,0.100,0.000"})
    
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Model Work Flow:**
        ```python
            classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        labels = ["bussiness","technology","science","sports","media and entertainment","politics","health","crime","accident","environment","art","literature","tragedy","education","fashion","food","travel","military","real estates","history","religion","celebrity"]
        hypothesis_template = "this text is about {}"

        results = []
        batch_size = 16
        for i in range(0, len(strings_list), batch_size):
            batch = strings_list[i:i+batch_size]
            predictions = classifier(batch, labels, hypothesis_template=hypothesis_template, multi_class=True)
            for j, prediction in enumerate(predictions):
                top_label = prediction['labels'][0]
                top_score = prediction['scores'][0]
                row_data = (dataFrame.iloc[i+j]['date'], dataFrame.iloc[i+j]['traffic'], dataFrame.iloc[i+j]['newsSnippet'], top_label, top_score)
                results.append(row_data)
        ```
        """
        )

    st.subheader("3.1.1.1 - Output examples")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(initial_classification_output_example1)

    with col2:
        st.write(initial_classification_output_example2)

    with col3:
        st.write(initial_classification_output_example3)

    space()

    st.subheader("3.1.2 - Category Clustering")
    st.markdown("In the first classification process, the data was classified into 22.")
    st.markdown("However, the number of categories was large for the relatively small amount of data, and therefore the data belonging to each category was too limited to be fully utilized. Therefore, I decided to group the categories into more general categories, thus grouping similar categories and the data belonging to them under one general category that represents all the categories to which they belong.")
    space()
    st.markdown("The category grouping process was carried out as follows:")
    
    st.table(
        {
            "Cluster": ["Economy", "Technology and Science", "Entertainment", "Lifestyle", "Accident", "Geopolitical", "Intellectualism"],
            "Includes": ["business, real estate", "technology, science", "media and entertainment, art, celebrity, sports", "environment, health, fashion, travel, food", "tragedy, crime, accident", "politics, military", "education, literature, history, religion"],
        }
    )
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Clusterin the detailed categories:**
        ```python
        data = pd.read_csv('./classified_data/regions/{some_region}/classified_data/{some_country}_classification_output.csv')
        
        label_cluster_mapping = {
            "business": "Economy",
            "real estates": "Economy",
            "technology": "Technology and Science",
            "science": "Technology and Science",
            "media and entertainment": "Entertainment",
            "arts": "Entertainment",
            "sports": "Entertainment",
            "celebrity": "Entertainment",
            "environment": "Lifestyle",
            "health": "Lifestyle",
            "fashion": "Lifestyle",
            "travel": "Lifestyle",
            "food": "Lifestyle",
            "tragedy": "Accident",
            "crime": "Accident",
            "accident": "Accident",
            "politics": "Geopolitical",
            "military": "Geopolitical",
            "education": "Intellectualism",
            "literature": "Intellectualism",
            "history": "Intellectualism",
            "religion": "Intellectualism",
        }
        
        df["general_label"] = df["predicted_label"].map(label_cluster_mapping)
        df.to_csv('./classified_data/regions/{some_region}/clustered_classified_data/{some_country}_clustered_classified.csv', index=False)
        ```
        """
        )
    
    st.subheader("3.1.2.1 - Output examples")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(clustered_classification_output_example1)

    with col2:
        st.write(clustered_classification_output_example2)

    with col3:
        st.write(clustered_classification_output_example3)
    
    st.markdown("Now, that how a topic-country trend might look like:")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/initial_classification_output_visual.png")

    space()

    st.subheader("3.1.3 - Preprocessing the classified data")
    st.markdown("To move to the next step, I had to reshape the data and make it simpler yet more understandable to use it efficiently.")
    st.markdown(
        """
            Here is how I reshaped the data:

            1. Normalized the values of topic-interaction rates.
            2. Deleted the unnecessary columns.
            3. The dates were set, and the data was arranged in chronological order daily.
            4. Sub-datasets were distributed based on their subject and country of origin.
                """
    )
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Normalization:**
        ```python
            def calculate_traffic_rate(value, max):
        if max == 0: 
            return 0.0
        rate = float(value / max)
        epsilon = 1e-9  
        if rate <= 0 + epsilon:
            return 0.0
        elif rate < 0.25 - epsilon:
            return 0.1
        elif rate < 0.5 - epsilon:
            return 1/4
        elif rate < 0.75 - epsilon:
            return 1/2
        else:  
            return 1.0
        
        maxTraffic = df['traffic_numeric'].max()
        df['traffic_rate'] = df['traffic_numeric'].apply(lambda x: calculate_traffic_rate(x, maxTraffic))
        
        specific_category = clusters[n]
        specific_category_data = category_time_distribution[
        category_time_distribution['general_label'] == specific_category]
        ```
        """
        )
    
    st.subheader("3.1.3.1 - Output examples")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(clustered_classified_data_with_relative_traffic_rates1)

    with col2:
        st.write(clustered_classified_data_with_relative_traffic_rates2)

    with col3:
        st.write(clustered_classified_data_with_relative_traffic_rates3)

    st.markdown("Here is the visualization of normalized data:")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/general_labeled_data_with_relative_traffice_rate_visual_example.png")

    space()
    
    st.subheader("3.2 - Region Grouping")
    st.markdown(
        """
        In the early stages of the project, I anticipated a lack of training data for the prediction model I wanted to train in the later stages of the project. Therefore, I considered the possibility of grouping several countries into a single region and using all the data from each region to predict the future performance of the entire region. I also implemented various procedures to collect as much data as possible under a specific category or country.
        At this stage, I thought the time was right to group the countries into regions. To group the countries, I needed to find a specific criterion to base my work on. So, I decided to calculate the similarity score between the countries' behaviors, compare them, and group the most similar countries together into a single region.
        I conducted this process in two stages. In the first stage, I tried to calculate the similarity scores by directly comparing interaction values, but I got poor results because the method of calculating similarity did not fully align with my idea of calculating the similarity score in overall interaction performance trends.
        """
    )
    
    st.subheader("3.2.1 - First similarity calculation method")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Calculating similarity by direct comparison:**
        ```python
        def compare_traffic_rates(file1_path, file2_path):
            try:
                with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
                    reader1 = csv.reader(file1)
                    reader2 = csv.reader(file2)
                    next(reader1)  
                    next(reader2)  
                    similarity_score = 0
                    total_comparisons = 0
                    for row1, row2 in zip(reader1, reader2):
                        if len(row1) < 3 or len(row2) < 3: 
                            print("Error: one of the rows does not have enough columns")
                            return None
                        try:
                            rate1 = float(row1[2])
                            rate2 = float(row2[2])
                        except ValueError:
                            print("Error: Could not convert rate to a number")
                            return None
                        total_comparisons += 1
                        if rate1 == rate2:
                            similarity_score += 1
                    return similarity_score, total_comparisons
            except FileNotFoundError:
                print("Error: One or both files not found.")
                return None
        ```
        """
        )
    
    with st.expander("Intially calculated similiarty values"):
    
        st.json(
            {
    "africa": {
        "Economy": {
            "Kenya-Nigeria": 99.370,  
            "Kenya-SouthAfrica": 98.10,  
            "Nigeria-SouthAfrica": 97.47,
            "new-Kenya-Nigeria": 0,  
            "new-Kenya-SouthAfrica": 0,  
            "new-Nigeria-SouthAfrica": 0,
            "mean":98.313333,
            "new-mean":0,
            "empty_days_mean":156,
            # "eligibility":0.6301 deprecated
        },
        "Technology and Science": {
            "Kenya-Nigeria": 80.38,  
            "Kenya-SouthAfrica": 81.01,  
            "Nigeria-SouthAfrica": 79.75, 
            "new-Kenya-Nigeria": 36.80,  
            "new-Kenya-SouthAfrica": 36.66,  
            "new-Nigeria-SouthAfrica": 56.09,
            "mean":80.38,
            "new-mean":43.85,
            "empty_days_mean":133,
            # "eligibility":0.605, deprecated
        },
        "Entertainment": {
            "Kenya-Nigeria": 60.13,  
            "Kenya-SouthAfrica": 60.76,  
            "Nigeria-SouthAfrica": 59.49,
            "new-Kenya-Nigeria": 86.46,  
            "new-Kenya-SouthAfrica": 73.49,  
            "new-Nigeria-SouthAfrica": 69.82,
            "mean":60.126667,
            "new-mean":73.49,
            "empty_days_mean":24,
            # "eligibility":2.505, deprecated
        },
        "Lifestyle": {
            "Kenya-Nigeria": 70.89,  
            "Kenya-SouthAfrica": 58.23,  
            "Nigeria-SouthAfrica": 57.59,
            "new-Kenya-Nigeria": 42.95,  
            "new-Kenya-SouthAfrica": 34.78,  
            "new-Nigeria-SouthAfrica": 63.99,
            "mean":62.236667,
            "new-mean":47.24,
            "empty_days_mean":112,
            # "eligibility":0.556, deprecated
        },
        "Accident": {
            "Kenya-Nigeria": 55.70,  
            "Kenya-SouthAfrica": 48.10,  
            "Nigeria-SouthAfrica": 49.37,
            "new-Kenya-Nigeria":22.69,
            "new-Kenya-SouthAfrica":22.94,
            "new-Nigeria-SouthAfrica":19.66,
            "mean":51.056667,
            "new-mean":21.43,
            "empty_days_mean":92,
            # "eligibility":0.554, deprecated
        },
        "Geopolitical": {
            "Kenya-Nigeria": 72.78,  
            "Kenya-SouthAfrica": 63.92,  
            "Nigeria-SouthAfrica": 62.03,
            "new-Kenya-Nigeria": 22.87,  
            "new-Kenya-SouthAfrica": 32.86,  
            "new-Nigeria-SouthAfrica": 26.32,
            "mean":66.243333,
            "new-mean":27.35,
            "empty_days_mean":118,
            # "eligibility":0.561, deprecated
        },
        "Intellectualism": {
            "Kenya-Nigeria": 65.82,  
            "Kenya-SouthAfrica": 57.59,  
            "Nigeria-SouthAfrica": 57.59,
            "new-Kenya-Nigeria": 21.00,  
            "new-Kenya-SouthAfrica": 30.67,  
            "new-Nigeria-SouthAfrica": 48.69,
            "mean":60.333333,
            "new-mean":30.45,
            "empty_days_mean":118,
            # "eligibility":0.511, deprecated
        },
    },
    "europe": {
        "Economy": {
            "Denmark-Finland": 88.61,  
            "Denmark-UK": 92.41,  
            "Finland-UK": 93.67,
            "new-Denmark-Finland": 0.51,  
            "new-Denmark-UK": 0,  
            "new-Finland-UK": 0,
            "mean":91.563333,
            "new-mean":0.17,
            "empty_days_mean":151,
            # "eligibility":0.605, deprecated
        },
        "Technology and Science": {
            "Denmark-Finland": 56.96,  
            "Denmark-UK": 53.80,  
            "Finland-UK": 50.63,
            "new-Denmark-Finland": 29.49,  
            "new-Denmark-UK": 31.58,  
            "new-Finland-UK": 73.00,
            "mean":53.796667,
            "new-mean":44.69,
            "empty_days_mean":81,
            # "eligibility":0.665, deprecated
        },
        "Entertainment": {
            "Denmark-Finland": 86.71,  
            "Denmark-UK": 77.85,  
            "Finland-UK": 73.42,
            "new-Denmark-Finland": 87.15,  
            "new-Denmark-UK": 87.46,  
            "new-Finland-UK": 79.39,
            "mean":79.326667,
            "new-mean":84.66,
            "empty_days_mean":2,
            # "eligibility":39.663, deprecated
        },
        "Lifestyle": {
            "Denmark-Finland": 48.10,  
            "Denmark-UK": 50.00,  
            "Finland-UK": 46.20,
            "new-Denmark-Finland": 42.69,  
            "new-Denmark-UK": 62.33,  
            "new-Finland-UK": 36.78,
            "mean":48.1,
            "new-mean":47.26,
            "empty_days_mean":46,
            # "eligibility":1.045, deprecated
        },
        "Accident": {
            "Denmark-Finland": 49.37,  
            "Denmark-UK": 51.90,  
            "Finland-UK": 60.76,
            "new-Denmark-Finland": 47.25,  
            "new-Denmark-UK": 55.11,  
            "new-Finland-UK": 56.66,
            "mean":54.01,
            "new-mean":53.00,
            "empty_days_mean":32,
            # "eligibility":1.688, deprecated
        },
        "Geopolitical": {
            "Denmark-Finland": 44.94,  
            "Denmark-UK": 48.73,  
            "Finland-UK": 43.67,
            "new-Denmark-Finland": 50.18,  
            "new-Denmark-UK": 56.91,  
            "new-Finland-UK": 38.63,
            "mean":45.78,
            "new-mean":48.57,
            "empty_days_mean":73,
            # "eligibility":0.627, deprecated
        },
        "Intellectualism": {
            "Denmark-Finland": 44.30,  
            "Denmark-UK": 50.00,  
            "Finland-UK": 46.20,
            "new-Denmark-Finland": 39.76,  
            "new-Denmark-UK": 43.84,  
            "new-Finland-UK": 37.81,
            "mean":46.833333,
            "new-mean":40.47,
            "empty_days_mean":60,
            # "eligibility":0.780, deprecated
        },
    },
    "north_america_australia": {
        "Economy": {
            "Australia-Canada": 94.94,  
            "Australia-USA": 96.84,  
            "Canada-USA": 95.57,
            "new-Australia-Canada": 0,  
            "new-Australia-USA": 0,  
            "new-Canada-USA": 0,
            "mean":95.783333,
            "new-mean":0,
            "empty_days_mean":154,
            # "eligibility":0.622, deprecated
        },
        "Technology and Science": {
            "Australia-Canada": 50.63,  
            "Australia-USA": 55.70,  
            "Canada-USA": 59.49,
            "new-Australia-Canada": 41.74,  
            "new-Australia-USA": 24.12,  
            "new-Canada-USA": 37.53,
            "mean":55.273333,
            "new-mean":34.46,
            "empty_days_mean":81,
            # "eligibility":0.682, deprecated
        },
        "Entertainment": {
            "Australia-Canada": 91.14,  
            "Australia-USA": 85.44,  
            "Canada-USA": 91.77,
            "new-Australia-Canada": 82.64,  
            "new-Australia-USA": 80.43,  
            "new-Canada-USA": 89.27,
            "mean":89.45,
            "new-mean":84.11,
            "empty_days_mean":2,
            # "eligibility":44.725, deprecated
        },
        "Lifestyle": {
            "Australia-Canada": 68.35,  
            "Australia-USA": 72.15,  
            "Canada-USA": 65.19,
            "new-Australia-Canada": 71.98,  
            "new-Australia-USA": 61.23,  
            "new-Canada-USA": 61.74,
            "mean":68.563333,
            "new-mean":64.98,
            "empty_days_mean":30,
            # "eligibility":2.285, deprecated
        },
        "Accident": {
            "Australia-Canada": 74.05,  
            "Australia-USA": 79.75,  
            "Canada-USA": 74.68,
            "new-Australia-Canada": 71.90,  
            "new-Australia-USA": 68.23,  
            "new-Canada-USA": 67.63,
            "mean":76.83,
            "new-mean":69.25,
            "empty_days_mean":11,
            # "eligibility":6.984, deprecated
        },
        "Geopolitical": {
            "Australia-Canada": 46.84,  
            "Australia-USA": 47.47,  
            "Canada-USA": 62.03,
            "new-Australia-Canada": 39.84,  
            "new-Australia-USA": 57.20,  
            "new-Canada-USA": 45.69,
            "mean":52.78,
            "new-mean":47.45,
            "empty_days_mean":51,
            # "eligibility":1.036, deprecated
        },
        "Intellectualism": {
            "Australia-Canada": 58.23,  
            "Australia-USA": 70.25,  
            "Canada-USA": 55.70,
            "new-Australia-Canada": 52.29,  
            "new-Australia-USA": 47.05,  
            "new-Canada-USA": 48.73,
            "mean":61.393333,
            "new-mean":49.35,
            "empty_days_mean":35,
            # "eligibility":1.754, deprecated
        },
    },
    "west_asia": {
        "Economy": {
            "Malaysia-Philippines": 95.57,  
            "Malaysia-Singapore": 93.67,  
            "Philippines-Singapore": 94.30,
            "new-Malaysia-Philippines": 3.93,  
            "new-Malaysia-Singapore": 17.89,  
            "new-Philippines-Singapore": 0,
            "mean":94.513333,
            "new-mean":7.27,
            "empty_days_mean":153,
            # "eligibility":0.618, deprecated
        },
        "Technology and Science": {
            "Malaysia-Philippines": 67.72,  
            "Malaysia-Singapore": 60.76,  
            "Philippines-Singapore": 63.29,
            "new-Malaysia-Philippines": 30.69,  
            "new-Malaysia-Singapore": 30.66,  
            "new-Philippines-Singapore": 30.45,
            "mean":63.923333,
            "new-mean":30.60,
            "empty_days_mean":104,
            # "eligibility":0.615, deprecated
        },
        "Entertainment": {
            "Malaysia-Philippines": 78.48,  
            "Malaysia-Singapore": 81.65,  
            "Philippines-Singapore": 91.77,
            "new-Malaysia-Philippines": 71.92,  
            "new-Malaysia-Singapore": 75.67,  
            "new-Philippines-Singapore": 95.28,
            "mean":83.966667,
            "new-mean":80.95,
            "empty_days_mean":12,
            # "eligibility":6.997, deprecated
        },
        "Lifestyle": {
            "Malaysia-Philippines": 62.66,  
            "Malaysia-Singapore": 52.53,  
            "Philippines-Singapore": 44.30,
            "new-Malaysia-Philippines": 63.58,  
            "new-Malaysia-Singapore": 57.39,  
            "new-Philippines-Singapore": 47.51,
            "mean":53.163333,
            "new-mean":56.16,
            "empty_days_mean":96,
            # "eligibility":0.554, deprecated
        },
        "Accident": {
            "Malaysia-Philippines": 52.53,  
            "Malaysia-Singapore": 49.37,  
            "Philippines-Singapore": 47.47,
            "new-Malaysia-Philippines": 43.98,  
            "new-Malaysia-Singapore": 39.52,  
            "new-Philippines-Singapore": 38.72,
            "mean":49.79,
            "new-mean":40.74,
            "empty_days_mean":79,
            # "eligibility":0.630, deprecated
        },
        "Geopolitical": {
            "Malaysia-Philippines": 76.58,  
            "Malaysia-Singapore": 55.06,  
            "Philippines-Singapore": 50.63,
            "new-Malaysia-Philippines": 17.05,  
            "new-Malaysia-Singapore": 24.84,  
            "new-Philippines-Singapore": 19.55,
            "mean":60.423333,
            "new-mean":20.48,
            "empty_days_mean":116,
            # "eligibility":0.522, deprecated
        },
        "Intellectualism": {
            "Malaysia-Philippines": 58.86,  
            "Malaysia-Singapore": 66.46,  
            "Philippines-Singapore": 55.06,
            "new-Malaysia-Philippines": 34.80,  
            "new-Malaysia-Singapore": 39.70,  
            "new-Philippines-Singapore": 49.23,
            "mean":60.793333,
            "new-mean":41.24,
            "empty_days_mean":85,
            # "eligibility":0.715, deprecated
        },
    }
            }
        )
    
    st.subheader("3.2.2 - Updated similarity calculation method")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Calculating similarity by calculating the cosine of the angle between compared vactors:**
        ```python
        def compute_similarity(df1, df2, label):
            df1_filtered = df1[df1['general_label'] == label]
            df2_filtered = df2[df2['general_label'] == label]
            
            common_dates = set(df1_filtered['date']).intersection(df2_filtered['date'])
            df1_filtered = df1_filtered[df1_filtered['date'].isin(common_dates)]
            df2_filtered = df2_filtered[df2_filtered['date'].isin(common_dates)]
            
            df1_filtered = df1_filtered.sort_values('date').reset_index(drop=True)
            df2_filtered = df2_filtered.sort_values('date').reset_index(drop=True)
            
            if len(df1_filtered) < 2 or len(df2_filtered) < 2:
                return None 
            
            traffic_rate_similarity = 1 - cosine(df1_filtered['traffic_rate'], df2_filtered['traffic_rate'])

            return {
                'traffic_rate_similarity': traffic_rate_similarity,
                'common_days': len(common_dates)
            }
        ```
        """
        )
    
    regions = list(cosine_similarity_values.keys())
    topics = list(next(iter(cosine_similarity_values.values())).keys())  # get topic list from any region

    selected_region = st.selectbox("🌍 Select Region", regions)
    selected_topic = st.radio("🧠 Select Topic", topics, horizontal=True)

    df = cosine_similarity_values[selected_region][selected_topic]
    st.dataframe(df)
    
    space(2)
    
    st.header("📍 CHAPTER 4: Handling the Lack of Data")

    st.subheader("4.1 - Fake Data Generator")
    st.markdown("The initial plan to solve the data shortage problem was to generate data similar to the original data and expand the database accordingly. However, initial experiments showed that this approach suffers from several drawbacks, such as difficulty in controlling the method of generating similar data while maintaining its similarity to reality, or generating data with patterns completely different from the original data.")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Attempting to generate orignal-alike fake data using this script:**
        ```python
        def generate_fake_data(real_data: pd.DataFrame, num_days: int) -> pd.DataFrame:

            real_data["date"] = pd.to_datetime(real_data["date"])

            start_date = real_data["date"].max() + timedelta(days=1)
            end_date = start_date + timedelta(days=num_days - 1)
            date_range = pd.date_range(start=start_date, end=end_date)

            synthetic_data = {
                "date": [],
                "general_label": [],
                "traffic_rate": [],
                "total_traffic": [],
            }

            general_label = real_data["general_label"].iloc[0]

            mean_traffic_rate = real_data["traffic_rate"].mean()
            std_traffic_rate = real_data["traffic_rate"].std()
            mean_total_traffic = real_data["total_traffic"].mean()
            std_total_traffic = real_data["total_traffic"].std()

            for date in date_range:
                seasonal_variation = np.sin(2 * np.pi * (date.dayofyear / 365))
                traffic_rate = np.clip(np.random.normal(mean_traffic_rate + seasonal_variation * 0.1, std_traffic_rate * 0.5),0,1,)
                total_traffic = np.clip(np.random.normal(mean_total_traffic * traffic_rate, std_total_traffic * 0.5),0,None,)

                synthetic_data["date"].append(date)
                synthetic_data["general_label"].append(general_label)
                synthetic_data["traffic_rate"].append(traffic_rate)
                synthetic_data["total_traffic"].append(total_traffic)

            synthetic_df = pd.DataFrame(synthetic_data)
            return synthetic_df
        ```
        """
        )
    
    st.subheader("4.2 - Google Trends")
    
    st.markdown("""
        After abandoning the previous idea, I searched for new potential solutions and discovered the "Google Trends", which emerged as a game-changing data source, offering highly relevant and structured information that closely mirrored my original dataset, thus enabling a seamless transition to more scalable and reliable modeling. meaning I didn't have to put in extra effort to process the imported data. In addition, the data scope is wide and diverse. However, I didn't start collecting data haphazardly. Instead, I first verified that Google's data was similar to the data I was working with in terms of performance and characteristics. I compared samples of the data I was working with to Google's data and concluded that Google's data was similar to a large portion of my data. Based on this, I made some adjustments, which I can summarize as follows:
        - I abandoned the division of countries into regions since there is ample data for each country and for any topic I wanted.
        - I modified the categories I had defined at the beginning of the project to make them consistent with the data categories on the Google Trends platform.
        - I used a vector similarity calculation method between the original data and the Google data.
        - Collecting data over a period of approximately five years, starting from 2012 and ending with the time period of the original data, in order to use the period of the original data in the process of testing the performance of the forecasting model that will be developed later.
                """)

    google_trend_data_sample['date'] = pd.to_datetime(google_trend_data_sample['date'])
    google_trend_data_sample.set_index('date', inplace=True)

    st.markdown("Google trends data sample where it represents the entertainment trend in USA for the last month:")

    st.line_chart(google_trend_data_sample)

    space(2)
    
    st.header("📍 CHAPTER 5: Transformers Time Series Forecasting Models")

    st.subheader("5.1 - First attampt: Keras Model")
    st.markdown("Using the Keras Sequential model with LSTM and Dense layers, I tried to train the Time Forecasting Model using only the original data, but the results were very poor.")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Model Setup:**
        ```python
        def create_model():
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(30, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        X_train_country1, y_train_country1 = preprocess_data(cointry_data1)
        model = create_model()
        model.fit(X_train_country1, y_train_country1, epochs=50, verbose=1)

        X_train_country2, y_train_country2 = preprocess_data(cointry_data2)
        model.fit(X_train_country2, y_train_country2, epochs=30, verbose=1)
        
        # I continued to tune the model using more tropic-country trned data.
        
        X_train_country12, y_train_country12 = preprocess_data(cointry_data12)
        model.fit(X_train_country12, y_train_country12, epochs=30, verbose=1)
        
        predictions, actual_data = test_on_time_range(
            model, 
            cointry_data4.set_index('date')['traffic_rate'], 
            start_date='2016-11-28', 
            end_date='2017-05-04'
        )
        ```
        """
        )
        
    st.subheader("5.1.1 - Output example")

    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/transformers_models/output.png")
    
    space()
    st.subheader("5.2 - Second attampt: Logic Update")
    st.markdown("I used the extended data later to train the same models. The results got better, but I hypothesized that more context-aware models might yield stronger performance, which led me to explore Prophet later.")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Model Update:**
        ```python
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data, seq_length)

        model = create_model(seq_length)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        predictions = model.predict(X_test)
        print("predictions:", predictions)

        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)
        ```
        """
        )
    st.subheader("5.1.2 - Output example")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/transformers_models/output3.png")
    
    space(2)
    
    st.header("📍 CHAPTER 6: Facebook Prophet Model")

    st.subheader("6.1 - Model Setup")
    st.markdown("After a quick research, I found out that the Facebook Prophet Time Series Forecasting model might be a good choice. So, I immediately followed a tutorial and implemented a basic logic to test the performance using the extended data.")
    
    with st.expander("Code Snippet"):
        st.markdown(
            """
        **Model Setup:**
        ```python
        train_subset = data.loc[data.index <= split_date].copy()
        test_subset = data.loc[data.index > split_date].copy()
        
        model = Prophet()
        model.fit(Pmodel_train_subset)
        ```
        """
        )

    st.subheader("6.1.1 - Output examples")
    example =initial_prophet_model_results.drop("with_regressors_MAE",axis=1)
    st.write(example)
    
    space()
    
    st.subheader("6.2 - Regressors")
    st.markdown("To enhance the performance of the model, I started to import data to enforce the context of the training dataset. This would make the model smarter by noticing unusual events instead of being only a past-data-based prediction model.")
    st.markdown("This process took me some time because it is not very straightforward. It requires general knowledge about the countries' interests and periodic events and requires me to make some assumptions, test them, and adjust them. However, it eventually paid off.")
    st.markdown("The chart below visualizes the **immigration** regressor, where I used it to check whether considering this data may enhance the **Geopolitical** trend interactivity forecasting in **USA** or not.")
    
    regressor_example['date'] = pd.to_datetime(regressor_example['date'])
    regressor_example.set_index('date', inplace=True)
    st.line_chart(regressor_example)
    
    
    st.subheader("6.2.1 - Output examples")
    st.markdown("Optimized performance after adding regressors")
    st.write(initial_prophet_model_results)
    space()
    
    st.subheader("6.3 - Country-Topic Trend Prediction Models")
    st.markdown("Here, I manually created 12 to 15 models that predict a topic trend behavior in some country. During this process, I took care of each model and optimized the models that had poor performance, and I succeeded in most cases.")
    
    with st.expander("Model Example"):
        st.markdown(
            """
        regressors = [
            {
                "label": "diplomacy",
                "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/diplomacy.csv",
            },
            {
                "label": "economy",
                "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/economy.csv",
            },
            {
                "label": "elections",
                "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/elections.csv",
            },
            {
                "label": "war",
                "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/war.csv",
            },
            {
                "label": "immigration",
                "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Geopolitical/extended_data/usa_trend_data/regressors/immigration.csv",
            },
        ]
        regressors_data = []

        for regressor in regressors:
            data = pd.read_csv(regressor["path"], parse_dates=["date"])
            data.set_index("date", inplace=True)
            data.rename(columns={"value": regressor["label"]}, inplace=True)
            regressors_data.append(data)

        country_trend_data = pd.read_csv(country_trend_data_path, parse_dates=["date"])
        country_trend_data.set_index("date", inplace=True)

        data = country_trend_data.join(
            regressors_data,
            how="left",
        )

        model = Prophet()
        for regressor in regressors:
            model.add_regressor(regressor["label"])
        model.fit(Pmodel_train_subset)
        ```
        """
        )
    st.subheader("6.3.1 - Output example")
    st.markdown("Below you can see the combination of the original topic-country trend data with the regressors data")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/output.png")

    st.subheader("6.3.2 - Output example")
    st.markdown("The fllowing visual shows train-test data split information with the predicted values")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/output3.png")
    st.warning("You can clearly see how distinct the trend data is, yet the prediction results are considered very good because the model has been supported with regressors to enhance the context. This means that the model is not only past-based forecasting model, but rather is an intelligent model.")
    
    st.subheader("6.3.3 - Output example")
    st.markdown("Here is a closer look at a single month trend forecasting.")
    st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/output4.png")
    
    
    st.subheader("6.3.4 - Output example")
    st.markdown("Take a look at the components that play a role in producing the final results.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/output5.1.png")
    with col2:
        st.image("C:/Users/alkrd/Desktop/graduation_project/the_project/TimeSeriesForecasting/facebook_prophet/output5.2.png")

main()
