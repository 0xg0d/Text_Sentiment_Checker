import argparse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", required=True, help="Path to the input file")
args = parser.parse_args()

# Set the input file path
input_file_path = args.input_file

# Open the input file and read the text
with open(input_file_path, "r") as input_file:
    text = input_file.read()

# Create the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Use the sentiment analyzer to analyze the text
sentiment_scores = analyzer.polarity_scores(text)

# Print the overall sentiment scores
print(sentiment_scores)

# Tokenize the text
tokens = word_tokenize(text)

# Remove stop words from the tokens
filtered_tokens = [word for word in tokens if word not in stopwords.words("english")]

# Print the filtered tokens
print(filtered_tokens)

# Use the sentiment analyzer to analyze the filtered tokens
filtered_sentiment_scores = analyzer.polarity_scores(" ".join(filtered_tokens))

# Print the filtered sentiment scores
print(filtered_sentiment_scores)

# Compute the average sentiment score for each sentence
sentences = text.split(".")
sentence_sentiment_scores = [analyzer.polarity_scores(sentence) for sentence in sentences]
average_sentiment_scores = [sum(sentiment_scores.values()) / len(sentiment_scores) for sentiment_scores in sentence_sentiment_scores]

# Print the average sentiment scores
print(average_sentiment_scores)
