import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 

tagalog_sentiment = {
  'lupet': 3,
  'galing': 3.2,
  'ayos': 2.7,
  'maganda': 3.7,
  'ganda': 3.4,
  'ayos': 3.2,
  'maayos': 3.5,
  'angas': 2.8,
  'panalo': 3.5,
  'mabilis': 3.2,
  'bilis': 2.8,
  'pangit': -3.4,
  'di ko gusto': -3.5,
  'di gusto': -3.3,
  'panget': -3.4,
  'ayaw': -3.1,
  'malungkot': -2.8,
  'lungkot': -2.8,
  'tapon': -3.4,
  'tinapon': -3.2
}

sia = SentimentIntensityAnalyzer()
sia.lexicon.update(tagalog_sentiment)

amazon_reviews = pd.read_csv('amazon_review.csv')
shopee_reviews = pd.read_csv('clean_extended_train.csv')
lazada_reviews = pd.read_json('reviews.json')

# print(amazon_reviews.info())
# print(shopee_reviews.info())

overall_reviews = pd.concat([amazon_reviews['reviewText'], shopee_reviews['review'].head(5000), lazada_reviews['review']], ignore_index=True)
sentiment_score = overall_reviews.astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

sentiments = pd.DataFrame({
  'sentiment_score': sentiment_score,
  'ratings': pd.concat([amazon_reviews['overall'], shopee_reviews['rating'].head(5000), lazada_reviews['rating']], ignore_index=True)
})

# print(sentiments.shape)
sentiments['repeat_buy'] = 0
print(sentiments.info())

sentiments.loc[
  (sentiments['sentiment_score'] >= 0.05) & (sentiments['ratings'] >= 4),
  'repeat_buy'
] = 1 #likely to buy again

sentiments.loc[
  (sentiments['sentiment_score'] <= -0.05) & (sentiments['ratings'] <= 2),
  'repeat_buy'
] = 0 #less likely to buy again

sentiments.loc[
  (sentiments['sentiment_score'].between(-1.75, 1.75)) & (sentiments['ratings'] == 3),
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
  # class_weight='balanced',
  # max_iter=1000,
  solver='lbfgs'
)

regression.fit(x_train_scaler, y_train)

y_result = regression.predict(x_test_scaler)
print("Accuracy: ", accuracy_score(y_test, y_result))
print(classification_report(y_test, y_result, zero_division=0))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_result))

# print(regression.coef_[0])

print('Coefficients: ')
for columns, coefficient in zip(x.columns, regression.coef_[0]):
  print(f"{columns}: {coefficient}")

y_result = regression.predict(x_test_scaler)
accuracy = accuracy_score(y_test, y_result)
print(f"Accuracy of the model: {accuracy:.2%}")

print("\n--- Monte Carlo Simulation ---")

simulations = 1000
simulated_results = []

start = time.time()

for _ in range(simulations):
    random_sentiment = np.random.uniform(-1, 1)
    random_rating = np.random.randint(1, 6)     

    sim_point = pd.DataFrame([[random_sentiment, random_rating]], columns=['sentiment_score', 'ratings'])

    sim_point_scaled = scaler.transform(sim_point)

    prediction = regression.predict(sim_point_scaled)[0]
    simulated_results.append((random_sentiment, random_rating, prediction))

print("Sample Monte Carlo Results (first 10):")
for i in range(10):
    sentiment, rating, result = simulated_results[i]
    label = {0: "Less likely to buy again", 1: "Likely to buy again", 2: "Unsure Buyer"}[result]
    print(f"Sentiment: {sentiment:.2f}, Rating: {rating}, Prediction: {label}")

count_0 = sum(1 for _, _, r in simulated_results if r == 0)
count_1 = sum(1 for _, _, r in simulated_results if r == 1)
count_2 = sum(1 for _, _, r in simulated_results if r == 2)

print("\nSimulation Summary:")
print(f"Less likely to buy again: {count_0}")
print(f"Likely to buy again: {count_1}")
# print(f"50/50: {count_2}")

end = time.time()

print('Execution Speed(1000): ', end-start)

h = 0.01
x_min, x_max = x['sentiment_score'].min() - 0.1, x['sentiment_score'].max() + 0.1
y_min, y_max = x['ratings'].min() - 0.5, x['ratings'].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = regression.predict(grid_scaled)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

plt.scatter(x_test['sentiment_score'], x_test['ratings'], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sentiment Score')
plt.ylabel('Ratings')
plt.title('Logistic Regression Decision Boundary')
plt.colorbar(label='Repeat Buy (0 = No, 1 = Yes)')
plt.grid(True)
plt.show()

monte_carlo_df = pd.DataFrame(simulated_results, columns=['sentiment_score', 'rating', 'repeat_buy'])

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=monte_carlo_df,
    x='sentiment_score',
    y='rating',
    hue='repeat_buy',
    palette={0: 'red', 1: 'blue', 2: 'black'},
    alpha=0.6,
    edgecolor='black'
)
plt.title('Monte Carlo Simulation Results')
plt.xlabel('Sentiment Score')
plt.ylabel('Rating')
plt.legend(title='Repeat Buy (0 = No, 1 = Yes)')
plt.grid(True)
plt.show()