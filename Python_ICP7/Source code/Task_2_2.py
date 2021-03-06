# importing Natural Language Toolkit
import nltk

# importing libraries
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, ngrams

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')

# fetching the data from input.txt file
input = open('in_150lines.txt', encoding="utf8").read()

# Tokenization
print("\n###========== Tokenization ==========### \n")

# tokenizing the input data into Sentences and Words
st_tokens = nltk.sent_tokenize(input)  # retrieve Sentence Tokens
wd_tokens = nltk.word_tokenize(input)  # retrieve Words Tokens

print("\n Tokenization: \n")
print("No.of Sentences: ", len(st_tokens))
print("No.of Words: ", len(wd_tokens))

print("\n###========== POS ==========### \n")

# performing Part of Speech tagging by using pos_tag
print("Part Of Speech output: \n")
print(nltk.pos_tag(wd_tokens))

# Stemming
print("\n###========== Stemming ==========### \n")

# PorterStemmer
pStemmer = PorterStemmer()
print("\nPorter Stemmer output: \n")
for i in st_tokens:
    print(pStemmer.stem(i), end='')

# LancasterStemmer
lStemmer = LancasterStemmer()
print("\nLancaster Stemmer output : \n")
for i in st_tokens:
    print(lStemmer.stem(i), end='')

# SnowballStemmer
sStemmer = SnowballStemmer('english')
print("\nSnowball Stemmer output : \n")
for i in st_tokens:
    print(sStemmer.stem(i), end='')

# Lemmatization
print("\n###========== Lemmatization ==========### \n")

# Applying Lemmatization using WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("\nLemmatization output : \n")
for i in st_tokens:
    print(lemmatizer.lemmatize(i), end=' ')

# Trigram
print("\n###========== Trigram ==========### \n")
print("\nTrigram output: \n")
token = nltk.word_tokenize(input)
n = 0
for s in st_tokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)

# Named Entity Recognition
print("\n###========== Named Entity Recognition ==========### \n")
print("\nNamed Entity Recognition output: \n")
n = 0
for s in st_tokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(nltk.word_tokenize(s))))