#! /usr/bin/env python
#encoding:UTF-8
'''
Created on 28 october 2011

@author: xavier HINAUT
xavier.hinaut@inserm.fr
'''

from __future__ import division

def remove_string_from_list_words(sentence, string):
    """
    Remove the given string from a list of words.
    
    Example :
    Inputs:
        sentence = ['<sil>', 'fill', 'may', 'have', 'a', 'ball', 'and', 'teeth', 'one', 'minute', 'common', 'next', '<sil>', 'azer', '<sil>', '<sil>', 'azer', '<sil>']
        string = '<sil>'
    (after processing):
        sentence = ['fill', 'may', 'have', 'a', 'ball', 'and', 'teeth', 'one', 'minute', 'common', 'next', 'azer', 'azer']
        
    """
    while 1:
        try:
            sentence.remove(string)
        except Exception:
#            print "removing finished"
            break
#        print "removing"
    return sentence

def l_has_words(l_ref, l_test):
    """
    makes two lists, one with the elements that are in l_test and l_ref, and one with the words present in l_test but not in l_ref
    """
    in_list = []
    not_in_list = []
    for w in l_test:
        if w in l_ref:
            in_list.append(w)
        else:
            not_in_list.append(w)
    return (in_list, not_in_list)

def string_to_l_word_sentence(l_string_sent):

    """
    Inputs:
        l_string_sent: list of sentences in the form of a list of strings
    Outputs:
        lws: list of sentences in the form of a list of list of words
    """
#    lss = copy.deepcopy(l_string_sent)
    lss = l_string_sent
    lws = []
    for ss in lss:
        new_s = ss.split(' ')
        lws.append(new_s)
    return lws

def extract_words_from_sent(sentences, l_words=[]):
    """
    inputs:
        - sentences: list of sentences in the form of a list of list of word (l_word_sentence)
        - l_words: a list of words
        
    output:
        - l_words: a list of words
    """
    for s in sentences:
        for w in s:
            if w not in l_words:
                l_words.append(w)
    return l_words

def get_sub_dic_from(d_words, l_words):            
    """
    Gets the sub-dictionary taking d_words
    
    inputs:
        - d_words: each key is a word, the corresponding value is not processed (it remain unchanged).
        - l_words: a list of words
        
    output:
        - new_d: new dictionary containing the words of sentences
        - words_not_in_dic: list containing the list of words that were not found in the dictionary
    """
    new_d = {}
    words_not_in_dic = []
    for w in l_words:
        if d_words.has_key(w):
            new_d[w] = d_words[w]
        else:
            words_not_in_dic.append(w)
    
    if words_not_in_dic!=[]:
        print "Some of the words were not found in the dictionary."
        print "Words not in dictionary: "+str(words_not_in_dic)
#        raise Warning, "Some of the words were not found in the dictionary."
    return (new_d, words_not_in_dic)

def empty_string(s):
    if s=='':
        return True
    elif s==' ':
        return True
#    elif s==" ":
#        return True
    else:
        return False
    
def count_nr_words(l_data):
    """
    Counts the number of words for each sentence, and return them as a list.
    
    inputs:
        - l_data: is a list of sentences. Each sentence is a list of words.
    """
    l_res = []
    for s in l_data:
        l_res.append(len(s))
    return l_res

def get_english_stopwords():
    """
    Stopwords are common words that generally do not contribute to the meaning of a sentence,
    at least for the purposes of information retrieval and natural language processing.
    """
    from nltk.corpus import stopwords
    english_stops = set(stopwords.words('english'))
    l = list(english_stops)
    l.sort()
    return l

def get_most_common_item_in_list(liste):
    l = liste[:]
    l.sort()
    last_item = l[0]
    nr_occ = 0
    most_common = None
    most_common_occ = 0
    for item in l:
#        if nr_occ==0:
#            nr_occ = 1
#            last_item = item
        if last_item==item:
            nr_occ += 1
            if nr_occ>most_common_occ:
                most_common = item
                most_common_occ = nr_occ
        else:
            nr_occ = 1
            last_item = item
    return (most_common, most_common_occ)

