import nltk
nltk.download('punkt')#token
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import wordcloud
from wordcloud import WordCloud
from matplotlib import pyplot as plt

#----------------------------#

my_corpus = nltk.corpus.PlaintextCorpusReader('C:/Users/Ronaldo/Documents/CienciaDeDados/aula4', r'.*\.txt')
 
#----------------------------#

my_corpus.words()
my_corpus.raw()

myTokens = word_tokenize(my_corpus.raw())
print(myTokens)

#----------------------------#

print(stopwords.words("portuguese"))
stopwords = stopwords.words("portuguese")
noStopwords = [t for t in myTokens if t not in stopwords]

#----------------------------#

punctuation = list(punctuation)
print(punctuation)
noPunctuation = [t for t in noStopwords if t not in punctuation]

#----------------------------#

number = ['0', '1','2','3','4','5','7','8','9']
noNum = [t for t in noPunctuation if t not in number]

#----------------------------#

twords = [w.upper() for w in noNum]

#----------------------------#

wc = WordCloud(background_color = 'black', max_words = 500, max_font_size = 50).generate(str(twords))

fig = plt.figure(1,figsize=(15,15))
plt.axis('off')
plt.imshow(wc, interpolation = 'bilinear')
plt.show()