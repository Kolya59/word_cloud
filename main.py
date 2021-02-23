import sys
import time
import json
import random
import string
from collections import Counter
from typing import List
import pymorphy2
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

sw = set(stopwords.words('russian'))
table = str.maketrans(dict.fromkeys(string.punctuation))
stopgramms = ['NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ']
morph = pymorphy2.MorphAnalyzer(lang='ru')


# Read data from file
def read_json(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


# Get clear data from json
def process(source: List[dict]) -> List[list]:
    return [normalize(read_msg(msg)) for msg in source if is_valid(msg)]


# Get chat with user from file
def get_data_from_person(source: List[dict], user_id: int) -> List[dict]:
    for item in source:
        if type(item) is dict and item['id'] == user_id:
            return item['messages']
    return []


# Validate msg
def is_valid(obj: dict) -> bool:
    if obj.get('type') != 'message':
        return False
    mb_text = obj.get('text')
    return type(mb_text) is str and mb_text != ''


# Read text from msg
def read_msg(obj: {}) -> str:
    return obj.get('text')


# Remove wasted words from msg
def normalize(text: str) -> [str]:
    text_without_punct = text.translate(table)
    tokens = []
    for w in word_tokenize(text_without_punct):
        p = morph.parse(w)[0]
        if p.tag.POS in stopgramms:
            continue
        tokens.append(w)
    filtered_sentence = [w.lower() for w in tokens if not w.lower() in sw and not w.isdigit()]
    return filtered_sentence


# Flat list
def flat(data: [[str]]) -> [str]:
    return [item for sub in data for item in sub if len(item) != 0]


# Generate word cloud from list of words
def generate_cloud(data: [str], out_path: str) -> None:
    freq = Counter(data)

    wc = WordCloud(width=1920, height=1080, max_words=20000, stopwords=set(STOPWORDS), margin=10,
                   random_state=1).generate_from_frequencies(freq)
    plt.title("Chat Word Cloud")
    plt.imshow(wc.recolor(random_state=int(random.random() * 256)), interpolation="bilinear")
    wc.to_file(out_path)
    plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    user_id = int(sys.argv[1])
    source = read_json('data.json')
    user_chat = get_data_from_person(source, user_id)
    processed_data = process(user_chat)
    generate_cloud(flat(processed_data), './res' + str(user_id) + '.png')
    stop = time.perf_counter()
    print(f"Done for {start - stop:0.4f} seconds")

