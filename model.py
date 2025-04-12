import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import time

sia = SentimentIntensityAnalyzer()

amazon_reviews = pd.read_csv('amazon_review.csv')
shopee_reviews = pd.read_csv('clean_extended_train.csv')

# print(amazon_reviews.info())
# print(shopee_reviews.info())

overall_reviews = pd.concat([amazon_reviews['reviewText'], shopee_reviews['review'].head(5000)], ignore_index=True)
sentiment_score = overall_reviews.astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

sentiments = pd.DataFrame({
  'sentiment_score': sentiment_score,
  'ratings': pd.concat([amazon_reviews['overall'], shopee_reviews['rating'].head(5000)], ignore_index= 5000)
})

sentiments['repeat_buy'] = 0

sentiments.loc[
  (sentiments['sentiment_score'] >= 0.05) & (sentiments['ratings'] >= 4),
  'repeat_buy'
] = 1 #likely to buy again

sentiments.loc[
  (sentiments['sentiment_score'] <= -0.05) & (sentiments['ratings'] <= 2),
  'repeat_buy'
] = 0 #less likely to buy again

sentiments.loc[
  (sentiments['sentiment_score'].between(-0.05, 0.05)) & (sentiments['ratings'] == 3),
  'repeat_buy'
] = 2 #50/50

# print(sentiments)

sentiments = sentiments[sentiments['repeat_buy'].notna()]

x = sentiments[['sentiment_score', 'ratings']]
y = sentiments['repeat_buy']

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size= 0.2, random_state= 42
)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

regression = LogisticRegression(
  multi_class='multinomial',
  max_iter=1000,
  solver='lbfgs'
)

regression.fit(x_train_scaler, y_train)

y_result = regression.predict(x_test_scaler)
print("Accuracy: ", accuracy_score(y_test, y_result))
print(classification_report(y_test, y_result, zero_division=0))

# print(regression.coef_[0])

print('Coefficients: ')
for columns, coefficient in zip(x.columns, regression.coef_[0]):
  print(f"{columns}: {coefficient}")

y_result = regression.predict(x_test_scaler)
accuracy = accuracy_score(y_test, y_result)
print(f"Accuracy of the model: {accuracy:.2%}")
