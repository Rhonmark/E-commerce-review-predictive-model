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

simulations = 10000
start = time.time()

mc_results = []

original_coefficients = regression.coef_[0]
original_intercept = regression.intercept_[0]

coefficient_std = 0.1
intercept_std = 0.1

n_test_points = 2000
test_sentiments = np.random.uniform(-1, 1, n_test_points)
test_ratings = np.random.randint(1, 6, n_test_points)
test_data = pd.DataFrame({
    'sentiment_score': test_sentiments,
    'ratings': test_ratings
})
test_data_scaled = scaler.transform(test_data)

all_predictions = np.zeros((simulations, n_test_points))
  
for sim in range(simulations):
    sampled_coefficients = np.random.normal(
        original_coefficients, 
        coefficient_std, 
        size=original_coefficients.shape
    )
    sampled_intercept = np.random.normal(
        original_intercept, 
        intercept_std
    )
  
    mc_model = LogisticRegression()
    mc_model.classes_ = regression.classes_
    mc_model.coef_ = np.array([sampled_coefficients])
    mc_model.intercept_ = np.array([sampled_intercept])
    
    predictions = mc_model.predict(test_data_scaled)
    all_predictions[sim] = predictions
    
    if sim < 10:
        for i in range(3):
            mc_results.append({
                'simulation': sim,
                'sentiment_score': test_data.iloc[i]['sentiment_score'],
                'rating': test_data.iloc[i]['ratings'],
                'prediction': predictions[i]
            })

end = time.time()

results_df = pd.DataFrame(mc_results)

print("\nSample Monte Carlo Results:")
for i in range(min(10, len(results_df))):
    sim = results_df.iloc[i]['simulation']
    sentiment = results_df.iloc[i]['sentiment_score']
    rating = results_df.iloc[i]['rating']
    prediction = results_df.iloc[i]['prediction']
    label = {0: "Less likely to buy again", 1: "Likely to buy again", 2: "Unsure Buyer"}[prediction]
    print(f"Simulation {sim}, Sentiment: {sentiment:.2f}, Rating: {rating}, Prediction: {label}")

prob_class_0 = np.mean(all_predictions == 0, axis=0)
prob_class_1 = np.mean(all_predictions == 1, axis=0)
prob_class_2 = np.mean(all_predictions == 2, axis=0)

test_data['prob_class_0'] = prob_class_0
test_data['prob_class_1'] = prob_class_1
test_data['prob_class_2'] = prob_class_2

test_data['most_likely_class'] = test_data[['prob_class_0', 'prob_class_1', 'prob_class_2']].idxmax(axis=1)
test_data['most_likely_class'] = test_data['most_likely_class'].map({
    'prob_class_0': 0,
    'prob_class_1': 1,
    'prob_class_2': 2
})

test_data['uncertainty'] = -(
    test_data['prob_class_0'] * np.log2(test_data['prob_class_0'] + 1e-10) +
    test_data['prob_class_1'] * np.log2(test_data['prob_class_1'] + 1e-10) +
    test_data['prob_class_2'] * np.log2(test_data['prob_class_2'] + 1e-10)
)

print("\nMonte Carlo Simulation Summary:")
print(f"Less likely to buy again (Class 0): {np.sum(test_data['most_likely_class'] == 0)} points")
print(f"Likely to buy again (Class 1): {np.sum(test_data['most_likely_class'] == 1)} points")
print(f"Unsure Buyer (Class 2): {np.sum(test_data['most_likely_class'] == 2)} points")

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
plt.colorbar(label='Repeat Buy (0 = No, 1 = Yes, 2 = Unsure)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    test_data['sentiment_score'],
    test_data['ratings'],
    c=test_data['most_likely_class'],
    s=test_data['uncertainty'] * 100,
    alpha=0.6,
    cmap=plt.cm.Paired,
    edgecolor='black'
)

plt.xlabel('Sentiment Score')
plt.ylabel('Rating')
plt.title('Monte Carlo Logistic Regression Results with Uncertainty')
plt.colorbar(scatter, label='Predicted Class')
plt.grid(True)

sizes = [0.2, 0.5, 1.0]
labels = ['Low uncertainty', 'Medium uncertainty', 'High uncertainty']
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             label=labels[i], 
                             markerfacecolor='gray', 
                             markersize=np.sqrt(sizes[i] * 100)) 
                  for i in range(len(sizes))] 
plt.legend(handles=legend_elements, title='Prediction Uncertainty')

plt.show()

plt.figure(figsize=(10, 6))
coefficient_samples = np.random.normal(
    original_coefficients.reshape(-1, 1), 
    coefficient_std, 
    size=(len(original_coefficients), 1000)
)

feature_names = ['Sentiment Score', 'Ratings']
for i, feature in enumerate(feature_names):
    plt.subplot(len(feature_names), 1, i+1)
    sns.histplot(coefficient_samples[i], kde=True)  
    plt.axvline(original_coefficients[i], color='r', linestyle='--')
    plt.title(f'Distribution of {feature} Coefficient')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()