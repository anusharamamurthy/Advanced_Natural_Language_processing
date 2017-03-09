from nltk.corpus import brown
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams
from math import log, exp
import itertools


tokens, tags = zip(*brown.tagged_words())
tagCounter = Counter(tags)
tagTags = defaultdict(Counter)
tokenTags = defaultdict(Counter)
for token, tag in brown.tagged_words():
    tokenTags[token][tag] +=1

posBigrams = ngrams(tags, 2)

for tag1, tag2 in posBigrams:
    tagTags[tag1][tag2] += 1

#Logic discussed with Anup Bharadwaj
def calculate(tags):
    global tokenTags,sentenceTokens,tagTags
    token = tags[0].split('|')[0]
    tag = tags[0].split('|')[1]
    prob = tokenTags[token][tag] / tagCounter[tag]
    prob = log(prob)
    for i in range(1, len(tags)):
        curTag = tags[i].split('|')[1]
        prevTag = tags[i - 1].split('|')[1]
        curToken = tags[i].split('|')[0]
        val = (tokenTags[curToken][curTag] / tagCounter[curTag]) * (tagTags[prevTag][curTag] / tagCounter[prevTag])
        prob += log(val) if val != 0 else 0
    return prob

tag_to_calculate = []
all_tags = []
sentence = "time flies like an arrow"
sentenceTokens = sentence.split()
for tok in sentenceTokens:
    appendedTokenTags = [tok+"|"+t for t in tokenTags[tok]]
    all_tags.append(appendedTokenTags)

comb = list(itertools.product(*all_tags))

prob_values = []
for lst in comb:
    prob = calculate(lst)
    prob_values.append(prob)

ind = prob_values.index(max(prob_values))

print(comb[ind])
