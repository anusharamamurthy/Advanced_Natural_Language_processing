# The Following code is a collaboration with Anup Bharadwaj

from collections import Counter
from collections import defaultdict

initial_vector = [0.45,0.35,0.15,0.05]

pos_tags = ['DT', 'JJ', 'NN', 'VB']
sentence = "a myth is a female moth".split()


def tag_to_index(tag):
    return pos_tags.index(tag)


def word_to_index(word):
    return sentence.index(word)


transition_matrix = []
emission_matrix = []

transition_matrix.append([0.03,0.42,0.5,0.05])
transition_matrix.append([0.01,0.25,0.65,0.09])
transition_matrix.append([0.07,0.03,0.15,0.75])
transition_matrix.append([0.3,0.25,0.15,0.3])

emission_matrix.append([0.85,0.05,0.03,0.05])
emission_matrix.append([0.01,0.10,0.45,0.1])
emission_matrix.append([0.02,0.02,0.02,0.6])
emission_matrix.append([0.85,0.05,0.03,0.05])
emission_matrix.append([0.01,0.6,0.25,0.05])
emission_matrix.append([0.12,0.13,0.25,0.2])

alpha = defaultdict(Counter)
beta = defaultdict(Counter)


def forward_accumulator(n):
    global alpha
    i = 0
    for tag in pos_tags:
        alpha[1][tag] = initial_vector[i] * emission_matrix[0][tag_to_index(tag)]
        i += 1


    for j in range(2,n+1):
        for tag in pos_tags:
            for tag_2 in pos_tags:
                alpha[j][tag] += alpha[j-1][tag_2] * transition_matrix[tag_to_index(tag_2)][tag_to_index(tag)]
            alpha[j][tag] *= emission_matrix[j-1][tag_to_index(tag_2)]

    return alpha[n]


def backward_accumulator(n):
    global beta
    for tag in pos_tags:
        beta[len(sentence)][tag] = 1

    for j in range(len(sentence)-1,-1,-1):
        for tag in pos_tags:
            for tag_2 in pos_tags:
                beta[j][tag] += beta[j+1][tag_2] * transition_matrix[tag_to_index(tag)][tag_to_index(tag_2)]
            beta[j][tag] *= emission_matrix[j][tag_to_index(tag_2)]

    return beta[n]

print(forward_accumulator(4)['NN'])
print(forward_accumulator(3)['VB'])
print(forward_accumulator(1)['DT'])
print(backward_accumulator(4)['NN'])
print(backward_accumulator(2)['NN'])
