import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Classicalcrossover singer Jackie Evancho will sing The Star Spangled Banner at Donald Trumps inauguration ceremony on Friday January 20th The 16yearold who was the runnerup on 2010s Americas Got Talent competition broke the news")
for ent in doc.ents:
    print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))
print(nlp.pipe_labels['ner'])