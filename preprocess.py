import pandas as pd 
import re
import string

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)  # Fixed regex pattern
    # Remove usernames/reddit mentions
    text = re.sub(r"u\/[A-Za-z0-9]+", "", text)  # Fixed character class
    # Remove special characters and punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Remove extra spaces 
    text = re.sub(r"\s+", " ", text).strip()  # Fixed to replace with single space
    return text

def preprocess_csv(input_path, output_path):
    df = pd.read_csv(input_path)  # Fixed: was df.read_csv instead of pd.read_csv
    df = df[['text','label']]  # drop unused columns
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"[✅] Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_csv(r"D:\Mental_health_AI\data\reddit_mental_health.csv", 
                   r"D:\Mental_health_AI\data\reddit_cleaned.csv")  # Added .csv extensions and raw strings