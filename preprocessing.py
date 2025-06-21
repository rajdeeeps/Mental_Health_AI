import os
import pandas as pd
import re

#Loading all csv files from folder
data_dir = '/content/drive/MyDrive/Mental_health_AI/data/reddit_mental_health'
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

#Converting csv files into DataFrames
all_df = [pd.read_csv(f) for f in all_files]
df = pd.concat(all_df, ignore_index=True)

#dropping all columns except subreddit and post
df = df[['post','subreddit']]

#dropping NaN and missing values
df.dropna(inplace=True)
df = df[df['post'].str.strip() != '']

print(df['subreddit'].value_counts())

#enconding labels for subreddit
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['subreddit'])
df['label'].tolist()