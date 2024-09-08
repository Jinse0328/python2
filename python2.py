import requests
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def text_from(url):
    r = requests.get(url)
    return r.text

def text_analysis(text):
    w = word_tokenize(text)
    s = set(stopwords.words('english'))
    w_filtered = [word.lower() for word in w if word.lower() not in s and word.isalpha()]
    freq = FreqDist(w_filtered)
    return freq

def display(freq):
    df = pd.DataFrame(list(freq.items()), columns = ['Word', 'Number Repeated'])
    df = df.sort_values(by = 'Number Repeated', ascending = False)
    print("Top 15 repeated words: ")
    print(df.head(15))

def main():
    data_url = "https://en.wikipedia.org/wiki/Rutgers_University"
    data_text = text_from(data_url)
    fd = text_analysis(data_text)
    display(fd)

if __name__ == "__main__":
    main()