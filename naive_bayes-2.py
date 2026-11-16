#pip install numpy scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

text=["I love this movie","it was nice movie","i don't like it"]
lables=["positive","positive","negative"]

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(text)
model=MultinomialNB()
model.fit(x,lables)
test=["I love this"]
test_data=vectorizer.transform(test)
result=model.predict(test_data)

print("text:",test[0])
print("prediction:",result[0])
