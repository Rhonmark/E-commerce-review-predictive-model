import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
df = pd.read_csv("amazon_review.csv")
df_1 = pd.read_csv("clean_extended_train.csv")

concatenated_reviews = pd.concat([df['reviewText'], df_1['review'].head(5000)], ignore_index=True) 

sentiment_score = concatenated_reviews.astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

def score_metric(score):
  if score >= 0.05:
    return 'Good'
  elif score <= -0.05:
    return "Bad"
  else:
    return "Neutral"

sentiment_expression = sentiment_score.apply(score_metric)

df['sentiment_expression'] = sentiment_expression[:len(df)]
df_1['sentiment_expression'] = sentiment_expression[len(df):]

df['repeat_buy'] = None
df_1['repeat_buy'] = None

for buy_id, value in df.iterrows():
  if value['sentiment_expression'] == 'Good' and value['overall'] >= 4.0:
    df.at[buy_id, 'repeat_buy'] = 'Yes'
  elif value['sentiment_expression'] == 'Bad' and value['overall'] <= 2.0:
    df.at[buy_id, 'repeat_buy'] = 'No'
  elif value['sentiment_expression'] == 'Neutral' and value['overall'] == 3.0:
    df.at[buy_id, 'repeat_buy'] = 'Maybe'

for buy_id, value in df_1.iterrows():
  if value['sentiment_expression'] == 'Good' and value['rating'] >= 4:
    df_1.at[buy_id, 'repeat_buy'] = 'Yes'
  elif value['sentiment_expression'] == 'Bad' and value['rating'] <= 2:
    df_1.at[buy_id, 'repeat_buy'] = 'No'
  elif value['sentiment_expression'] == 'Neutral' and value['rating'] == 3:
    df_1.at[buy_id, 'repeat_buy'] = 'Maybe'

result_df = pd.concat([df[['repeat_buy']], df_1[['repeat_buy']].head(5000)], ignore_index=True)
valid_rows = len(result_df[result_df['repeat_buy'].notna()])
print(valid_rows)

likely = result_df[result_df['repeat_buy'] == 'Yes'].shape[0] / valid_rows * 100
less_likely = result_df[result_df['repeat_buy'] == 'No'].shape[0] / valid_rows * 100
maybe = result_df[result_df['repeat_buy'] == 'Maybe'].shape[0] / valid_rows * 100

print(f'Likely to repeat an order: {likely:.2f}%')
print(f'Less likely to repeat an order: {less_likely:.2f}%')
print(f'Not sure to repeat an order: {maybe:.2f}%')

print(likely + less_likely + maybe)
