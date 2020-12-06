#!/usr/bin/env/py35
# coding=utf-8
import re
import codecs
import pickle
import math
import codecs
import random
import logging
import numpy as np
import jieba
import sys
import jieba.posseg as pseg
import traceback
import MySQLdb
import os
import pyodbc
import pynlpir
import operator
import pymysql
import time

from random import random
from jieba.analyse import extract_tags,textrank,set_stop_words
from snownlp import SnowNLP
from datetime import datetime

#from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
jieba.initialize()


symbol_pun=["。", "？", "！","；"]

#!/usr/bin/env/py35
# -*-coding:utf-8-*-
# !/usr/bin/python3
# -*- coding:utf-8 -*-
import re
from numpy import log





def db_conn(**kwargs):
    conn = None
    try:
        conn = pymysql.connect(**kwargs)
    except Exception as e:
        print("db error:",e)

    return conn

def tokenize(sentense,restore=None):
    """
        @description: tokenize the sentense input by string form,and generate the tokens list form for word2vec model
        @params: description about company products
        @params:
        @return: [words...]
    """

    seg_word = []
    word_pos = {}
    seglist = pseg.cut(sentense)
    for seg in seglist:
        if (seg.flag not in ("x","uj",'r','y','e','o','un','ul','c') or len(seg.word) > 1) and (clean_text(seg.word) != "") and seg.word not in ("是",):
            #continue
        #if seg.word not in stop_words:

            seg_word.append(seg.word)
            word_pos[seg.word] = {}
            if seg.flag not in word_pos[seg.word]:
                word_pos[seg.word][seg.flag]=0
            word_pos[seg.word][seg.flag] += 1
    if restore:
        with open(restore,'wb') as pw:
            pickle.dump(word_pos,pw)
    return seg_word


def clean_text(text):
    """
        @params: input string format text
        @return: string text with no symbol
    """
    content = re.sub("\s+","",text)
    #content = re.sub("\t","",content)
    content = re.sub("\W+","",content)
    #content = re.sub("[0-9A-Za-z_]+","",content)
    #print(content)
    return content


def clean_stop(dictText,stopWords=[]):
    with open(path,'r') as f:
        for stopWord in f.readlines():
            stopWord = re.sub('\s+','',stopWord)
            stopWords.append(stopWord)
    print(stopWords[:5])
    dictText_cp = dictText.copy()
    for key in dictText:
        if key in stopWords:
            dictText_cp.pop(key)
    return dictText_cp

def ngarm(words,n,pmi=False):
    """
        @params: input string format text
        @params: input int n for dimonsion of sequence
        @return: {seq1:value,...}
    """

    #words = clean_text(text)
    #print("切分后的单词形式：")
    #print(words)
    wordDict = {}
    for i in range(len(words)+1-n):

        ngarm_seq = "".join(words[i:i+n])
        if pmi:
            ngarm_seq = " ".join(words[i:i+n])
        if ngarm_seq not in wordDict:
            wordDict[ngarm_seq] = 0
        wordDict[ngarm_seq] += 1
    # print("数据字典key信息：%s" %list(wordDict.keys())[:3])
    return wordDict


def garm_entroy(text,eplision = 1e-2,pmi_limit=12,freq_limit=4,ratio=0.8,printf=False,space_sub=True):
    """
        @params:
        @params:
        @params:
        @return:
    """
    words = tokenize(text)
    one_garm = ngarm(words,1,pmi=True)
    bin_garm = ngarm(words,2,pmi=True)
    tri_garm = ngarm(words,3,pmi=True)
    #print("\n"+'='*20)
    #print(tri_garm)

    bin_entroy = {}
    tri_score = {}
    score = {}
    one_values = sum(one_garm.values())
    bin_values = sum(bin_garm.values())
    tri_values = sum(tri_garm.values())

    for key in bin_garm:
        key_sp = key.split(' ')
        freq_word1 = one_garm[key_sp[0]]
        freq_word2 = one_garm[key_sp[1]]

        bin_prob = bin_garm[key]/float(bin_values)
        uni_freq_word1 = freq_word1 - bin_garm[key]
        uni_freq_word2 = freq_word2 - bin_garm[key]
        bin_entroy[key] = log(bin_prob*(one_values**2)/float(uni_freq_word1*uni_freq_word2+eplision))
        if bin_entroy[key] >= pmi_limit and bin_garm[key] >= freq_limit:
            score_tmp = bin_entroy[key]*bin_garm[key]
            #word_key = "".join(key_sp)
            score[key] = [bin_entroy[key],bin_garm[key],score_tmp]

    for key in tri_garm:
        #print(key)
        key_sp = key.split(' ')
        # print("三元键切分：%s" % key_sp)
        freq_word_r = bin_garm[" ".join(key_sp[:-1])]
        uni_freq_word_r = one_garm[key_sp[len(key_sp)-1]]

        freq_word_l = bin_garm[" ".join(key_sp[1:])]
        uni_freq_word_l = one_garm[key_sp[0]]

        uni_freq_word_m = one_garm[key_sp[1]]
        tri_prob = tri_garm[key]/float(tri_values)
        freq_a_word_r = uni_freq_word_r - tri_garm[key]
        freq_a_word_m = uni_freq_word_m - tri_garm[key]
        freq_a_word_l = uni_freq_word_l - tri_garm[key]
        tri_entroy = log(tri_prob*(one_values**3)/float(freq_a_word_l*freq_a_word_m*freq_a_word_r+eplision))
        freq_word_r_h = freq_word_r - tri_garm[key]
        freq_word_l_h = freq_word_l - tri_garm[key]
        tri_entroy_r = log(tri_prob*bin_values*one_values/float(freq_word_r_h*freq_a_word_r+eplision))
        tri_entroy_l = log(tri_prob*bin_values*one_values/float(freq_word_l_h*freq_a_word_l+eplision))
        bin_entroy_r = bin_entroy[" ".join(key_sp[:-1])]
        bin_entroy_l = bin_entroy[" ".join(key_sp[1:])]
        tri_score[key] = tri_entroy - min(tri_entroy_l,tri_entroy_r) + min(bin_entroy_l,bin_entroy_r)
        if (key == "文体 两 开花") and printf:
            print("tri_score=%3.f and tri_entroy=%.3f and  freq=%d" %(tri_score[key],tri_entroy,tri_garm[key]))
        if tri_entroy >= pmi_limit*1.5 and tri_garm[key] >= freq_limit:
            score_tmp = tri_score[key]*tri_garm[key]
            #word_key = "".join(key_sp)
            score[key] = [tri_score[key],tri_garm[key],score_tmp]
        if not score:
            continue
        if key in score:
            # print("score info key:{} and value={}".format(key,score[key]))
            bin_left = " ".join(key_sp[1:])
            bin_right = " ".join(key_sp[:-1])
            if (bin_left in score) and (score[bin_left][1] <= score[key][1]):
                score.pop(bin_left)
            if (bin_right in score) and (score[bin_right][1] <= score[key][1]):
                score.pop(bin_right)

    if space_sub:
        score_res = {}
        for key in score:
            new_key = re.sub(" ","",key)
            score_res[new_key] = score[key]
    else:
        score_res = score.copy()
    if printf:
        print("length to score_res:%d" %len(score_res))
        for _k in score_res:
            print("construct word: %s entroy= %.4f and freq= %d and score= %.2f" %(_k,score_res[_k][0],score_res[_k][1],score_res[_k][2]))
    return score_res

def compare_score(wordScore,ratio=0.8):
    """
        @description: compare the sorted words generated by function upon and
                    if keys which is different length exist relationship,compare their score and select the scorer one
        @params: wordScore input format-- [(key,[value]),...],ratio:double the thresholder to control the output record
        @return: new record with same format of input
    """
    i = 0
    index = list(range(len(wordScore)))
    index_len = len(wordScore)
    wordScore_cp = wordScore.copy()  #创建一个链表副本，用来存储新的表记录
    print("索引列表：%s" %index_len)
    for key_raw,value_ls in wordScore:
        key = key_raw.split(" ")
        if len(key) == 3:
            comp_key = [" ".join(key[:2])," ".join(key[1:])]
            print("\n输出临时比较键列表：(%d---%s)" %(i,comp_key))
            index.remove(i)
            #assert(isinstance(new_index,list))
            counter = 0
            for j in index:
                innerCounter = 0
                if wordScore[j][0] in comp_key:
                    print("输出第%s条索引记录结果,以及最后索引信息：%s" %(j,index))
                    innerCounter += 1
                    print("满足键关系的条数:%d" %innerCounter)
                    if wordScore[j][1][2] / value_ls[2] < 0.8:
                        wordScore_cp.remove(wordScore[j])
                    if value_ls[2]/wordScore[j][1][2] < 0.8:
                        wordScore_cp.remove((key, value_ls))
                    index.remove(j)
                    print("目前的索引：%s" %index)
                else:
                    continue
                counter += 1
            print("遍历的次数：%d" %counter)

            """
            for j in range(i):
                if wordScore[j][0] in comp_key:
                    if value_ls[2]/wordScore[j][1][2] < 0.8:
                        wordScore_cp.remove((key,value_ls))

            for j in range(i+1,index_len):
                if wordScore[j][0] in comp_key:
                    if wordScore[j][1][2]/value_ls[2] < 0.8:
                        wordScore_cp.remove(wordScore[j])
            """
        i += 1
    return wordScore_cp

def edit_dis(str1,str2):
    dis_cn=0
    if len(str1)<=len(str2):
        dis_cn += len(str2)-len(str1)
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                dis_cn += 1
    else:
        return edit_dis(str2,str1)
    return dis_cn

def is_confict(entroy_key,entroy_res):
    for _k in entroy_res.keys():
        if edit_dis(_k,entroy_key) == abs(len(entroy_key)-len(_k)):
            if entroy_res[entroy_key][1] < entroy_res[_k][1]:
                return False
    return True



def combineDegree(text,thereshold=20):
    """
        @description: input text and compute the ngrams limit the thereshold = 10 to limit the output result
        @params:*args--string format component of the corpus
        @params:
        @return: return dict contain words and freq like {word1:freq1...}
    """

    one_garm = ngarm(text,1)
    bin_garm = ngarm(text,2)
    tri_garm = ngarm(text,3)
    word_freq = {}
    for bin_word in bin_garm:
        word_sp = bin_word.split(' ')
        combine_prob = bin_garm[bin_word]/sum(bin_garm.values())
        prob_left = one_garm[word_sp[0]]/sum(one_garm.values())
        prob_right = one_garm[word_sp[1]]/sum(one_garm.values())
        prob_together = prob_left*prob_right
        if prob_together*thereshold < combine_prob:
            word_freq[bin_word] = bin_garm[bin_word]
        else:
            for word in word_sp:
                word_freq[word] = one_garm[word]
    for tri_word in tri_garm:
        word_sp = tri_word.split(" ")
        combine_prob = tri_garm[tri_word]/sum(tri_garm.values())
        prob_left_one = one_garm[word_sp[0]]/sum(one_garm.values())
        right_bin = ' '.join(word_sp[1:])



        prob_right_bin = bin_garm[right_bin]/sum(bin_garm.values())
        prob_right_one = one_garm[word_sp[2]]/sum(one_garm.values())
        left_bin = ' '.join(word_sp[:-1])
        prob_left_bin = bin_garm[left_bin]/sum(bin_garm.values())
        prob_middle = one_garm[word_sp[1]]/sum(one_garm.values())
        if prob_left_one*prob_right_one*prob_middle*thereshold < combine_prob or prob_left_one*prob_right_bin*thereshold < combine_prob or prob_right_one*prob_left_bin*thereshold < combine_prob:
            word_freq[tri_word] = tri_garm[tri_word]
            for bin_word in (right_bin,left_bin):
                if bin_word in tri_word and bin_word in word_freq:
                    word_freq.pop(bin_word)
            for word in word_sp:
                if word in tri_word and word in word_freq:
                    word_freq.pop(word)
    word_res = {}
    for key in word_freq:
        new_key = re.sub(" ","",key)
        word_res[new_key] = word_freq[key]
    return word_res


def construct_word_ngram(text,ngarm,n,thereshold,printf=False):
    """

    :param text:
    :param pos_words:
    :param ngarm:
    :param n:
    :param thereshold:
    :param printf:
    :return:
    """
    words = tokenize(text)
    one_garm = ngarm(words,1)
    word_freq = {_word:(one_garm[_word],len(_word)) for _word in one_garm}
    if printf:
        print("one garm length=%d" %len(one_garm))
    i = 2
    while i<=n:
        n_garm = ngarm(words,i)
        right_garm = ngarm(words,i-1)
        for _word_n in n_garm:
            word_sp = list(jieba.cut(_word_n))
            one_word = word_sp[0]

            right_word = "".join(word_sp[1:])
            #one_pos = pos_words[one_word]
            #convert_pos
            combine_prob = n_garm[_word_n]/sum(n_garm.values())
            if one_word in one_garm:
                one_prob = one_garm[one_word]/sum(one_garm.values())
            else:
                one_prob = 0.0
            if right_word in right_garm:
                right_prob = right_garm[right_word]/sum(right_garm.values())
            else:
                right_prob = 0.0
            if one_prob*right_prob*thereshold<combine_prob:
                word_freq[_word_n] = (n_garm[_word_n],len(_word_n))
            if right_word in word_freq:
                word_freq.pop(right_word)
            if one_word in word_freq:
                word_freq.pop(one_word)
        i += 1
    word_res = {}
    tmp_word = ""
    word_freq_flat = [(_key,_value[0],_value[1]) for _key,_value in word_freq.items()]
    word_freq_sorted = sorted(word_freq_flat, key=operator.itemgetter(1,2), reverse=True)
    count = 0
    for _word,_freq,_ in word_freq_sorted:

    #    if _word not in tmp_word:
    #        if count < 30 and printf:
    #            print("current id = %d and tmp_word:%s" %(count,tmp_word))

        word_res[_word] = _freq
        #tmp_word = _word
        #count += 1
    print("word num:",len(word_res))
    if printf:
        for idx,(word,freq,char_len) in enumerate(word_freq_sorted):
            print("top %d construct word [%s] display freq= %d and char_len=%d" %(idx+1,word,freq,char_len))
    return word_res

import traceback
lemma2id = pickle.load(open("lemma2id.pkl",'rb'))
def serials_construct(score_raw,pos_file,write2file=None,printf=False,insert2db=False):
    """
    save construct word with format word\tfreq\tpos
    :param score_res:
    :param pos_file:
    :param write2file:
    :param printf:
    :return:
    """
    if insert2db:
        db_info = dict(host="221.226.72.226", port=13306, user="root", passwd="somao1129", db="baidu", charset="utf8")
        conn = None
        while not conn:
            conn = db_conn(**db_info)
            time.sleep(2)
        cursor = conn.cursor()

    words_pos = pickle.load(open(pos_file,'rb'))
    count = 0
    if write2file:
        construct_write = open(write2file,'w',encoding="utf-8")
    construct_dict = {}
    for _raw in score_raw:
        words = _raw.split(" ")
        raw_pos = "x"
        try:
            first_pos = words_pos[words[0]]
            last_pos = words_pos[words[-1]]
            pre_last_pos = words_pos[words[-2]]
            if (("v" in last_pos or "vg" in last_pos) and ("v" not in pre_last_pos)) and ("vg" not in pre_last_pos)\
                    or (("a" in last_pos) and ("d" in pre_last_pos)) or \
                    (("v" in pre_last_pos or "vg" in pre_last_pos) and ("r" in last_pos))\
                    or (("n" in pre_last_pos) and ("a" in last_pos))\
                    or (("v" in pre_last_pos) and ("q" in last_pos or "qg" in last_pos))\
                    or (("v" in pre_last_pos) and ("f" in last_pos or "ug" in last_pos or "a" in last_pos or "uz" in last_pos or "l" in last_pos or "zg" in last_pos)):
                raw_pos = "v"
            elif ("n" in last_pos) or ("nz" in last_pos) or ("an" in last_pos) or ("nrt" in last_pos) \
                    or ("nr" in last_pos) or ("ns" in last_pos) or ("ng" in last_pos)\
                    or (("r" in pre_last_pos) and ("v" in last_pos))\
                    or (("v" in pre_last_pos) and ("d" in last_pos or "ad" in last_pos))\
                    or (("m" in pre_last_pos) and ("r" in last_pos or "m" in last_pos))\
                    or (("nr" in pre_last_pos) and ("z" in last_pos or "zg" in last_pos or "g" in last_pos))\
                    or (("t" in pre_last_pos) and ("t" in last_pos))\
                    or ("nrt" in pre_last_pos and "f" in last_pos)\
                    or (("vg" in pre_last_pos or "n" in pre_last_pos or "nr" in pre_last_pos or "ng" in pre_last_pos) and ("g" in last_pos))\
                    or ("nr" in pre_last_pos and "q" in last_pos)\
                    or ("n" in pre_last_pos and "ag" in last_pos):
                raw_pos = "n"
            elif words[0] != words[-2]:
                if (("n" in first_pos or "nz" in first_pos or "nr" in first_pos)) and (("v" in pre_last_pos or "nz" in pre_last_pos)):
                    raw_pos = "n"
                elif ("v" in first_pos and "v" in pre_last_pos):
                    pat_mul1 = re.compile(r'(.)(\1+)')
                    pat_mul2 = re.compile(r'(.{2})(\1+)')
                    sub_raw = re.sub(pat_mul1 or pat_mul2,"",_raw.replace(" ",""))
                    if len(sub_raw) == len(_raw.replace(" ","")):
                        raw_pos = 'n'
                elif (("a" in last_pos or "ag" in last_pos) and ("a" in pre_last_pos)):
                    raw_pos = "n"
        except:
            logger.info(traceback.format_exc())
            logger.info("no match word: %s" %_raw)

            continue
        word = _raw.replace(" ","")
        line = " ".join([word,str(score_raw[_raw][1]),raw_pos])
        if _raw.replace(" ","") not in construct_dict:
            construct_dict[word] = ["",""]
            construct_dict[word][0] = score_raw[_raw][1]
            construct_dict[word][1] = raw_pos
        else:
            construct_dict[word][0] += score_raw[_raw][1]
            if construct_dict[word][1] != "x":
                if raw_pos != "x" and score_raw[_raw][1]>construct_dict[word][0]:
                    construct_dict[word][1] = raw_pos
            else:
                if raw_pos != "x" or score_raw[_raw][1]>construct_dict[word][0]:
                    construct_dict[word][1] = raw_pos

        if word in lemma2id:
            word_id = lemma2id[word]
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql = "insert into construct_new_word (lemma_id,word,pos,timestamp) values ('%s','%s','%s','%s');" %(word_id,word,raw_pos,time_stamp)
            try:
                cursor.execute(sql)
                count += 1
            except:
                print(traceback.format_exc())
                logger.info("error record: %s" %word)
            logger.info("sucessful insert into %d records data into tb" %count)
        if printf and count<10:
            print("write to file line: %s" %line)
    conn.commit()
    if conn:
        cursor.close()
        conn.close()

    if write2file:
        for _key in construct_dict:
            m_flag = False
            if (construct_dict[_key][1] != "x" or construct_dict[_key][0] >= 7) and (len(_key) >= 2) and (len(_key) <= 5):
                for word,pos in pseg.cut(_key):
                    if pos in ("m","d","md","u","z","p","k"):
                        m_flag = True
                        break
                if m_flag:
                    continue
                line = " ".join([_key,str(construct_dict[_key][0]), construct_dict[_key][1]])
                construct_write.write(line+ "\r\n")


def clean_text(text):
    pattern = re.compile(r'([\w])(\1+)')
    #name = re.sub('\（.*\）', '', text)
    name = re.sub('\s+', '', text)
    #name = re.sub('\W+', '', name)
    name = re.sub('[a-zA-Z0-9_]+', '', name)
    name = re.sub(pattern,r'\1',name)
    return name

def split_sentences(text):
    sentences = {}
    sentence = ''
    id = 0
    for word in text:
        if word in symbol_pun:
            id += 1
            sentences.update({sentence:id})
            sentence = ''
        else:
            sentence += word
    return sentences

def genIDFfromTXT(files,write2file):
    words_dict = {}
    for file in files:
        with codecs.open(file,mode='r',encoding='utf-8') as fr:
            words_len = len(fr.readlines())

            for line in fr.readlines():
                line = line.rstrip()
                line = line.rstrip('\r\n')
                try:
                    word,freq,pos = line.split()
                except:
                    print("current line: has no enough words unpack" %line)
                    continue
                else:
                    if word not in words_dict:
                        words_dict[word] = log(words_len/freq)
    with codecs.open(write2file,'rb') as fw:
        pickle.dump(words_dict,fw)

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        #0x3000表示中文的全角空格符,是unicode的16进制表示(0x开头,8进制一般以0开头),对应的十进制是12288,32是英文半角空格的ascii(0-255),也是十进制的unicode值,通过chr还原成字符
        if num == 0x3000:
            num = 32
        #0xFF01,0xFF5E分别是中文字符信息'！','～'
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def half_to_full(s):
    """
    convert half-width charater to ful
    :param s:
    :return:
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 32:
            num = 0x3000
        elif 0x21 < num < 0x7E:
            num += 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def find_max_length(sentences):
    max_length=0
    for sentence in sentences:
        sentence = sentence.rstrip('\r\n')
        tmp=len(sentence)
        if tmp>max_length:
            max_length=tmp
    return max_length

class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

===============================================================================
# !/usr/bin/python2
# -*- coding:utf-8 -*-

# set_stop_words(path + 'stop_words_new.txt')

defaultEncoding = 'utf-8'
if sys.getdefaultencoding() != defaultEncoding:
    reload(sys)
    sys.setdefaultencoding(defaultEncoding)
jieba.load_userdict("D:/jwh/nlp/construct_word/construct_word_entroy.txt")


def stopWord(fpath):
    """
        @description: read the file from path load into the stopwords
        @params: exits the stopwords txt format file which line standford a stop word
        @params:
        @return: stopwords list []
    """
    stop_words = []
    with open(fpath, mode='r') as f:
        for line in f.readlines():
            word = re.sub("\s+", "", line.strip())
            stop_words.append(word)
    return stop_words

def tokenize(sentense,stop_words=[]):
    """
        @description: tokenize the sentense input by string form,and generate the tokens list form for word2vec model
        @params: description about company products
        @params:
        @return: [words...]
    """
    seg_word = []

    seglist = pseg.cut(sentense)
    for seg in seglist:
        if seg.flag not in ("nv", "v", "n", "x") or len(seg.word) == 1:
            continue
        if seg.word not in stop_words:
            seg_word.append(seg.word)
    return seg_word



def simToScore(sim):
    """
        @description: transform the sim result into score between [40,100]
        @params: sim--float type
        @return: float score
    """
    if sim >= 1:
        score = 95 + random() * 5
    elif sim < 0.16:
        score = 40 + random()
    else:
        score = sim ** 0.5 * 100 + random()
    if score > 100:
        score = 100
    return round(score, 2)


def getDistance(simhash1, simhash2):
    distance = 0
    this = "0b" + simhash1
    that = "0b" + simhash2
    n = int(this, 2) ^ int(that, 2)
    while n:
        n &= (n - 1)
        distance += 1
    return distance


def list2stringEncode(ls):
    """
        @description: transform the list to STRING with encode('utf-8') to display
        @params: list input with String element into unicode
        @return: STRING OUTPUT
    """
    if not ls:
        print "传入的是空列表"
    else:
        encode_ls = []
        for ele in ls:
            if not type(ele) == unicode:
                ele = ele.decode("utf")
            encode_ls.append(ele.encode("utf-8"))
        stringRes = ", ".join(encode_ls)
        return stringRes


def distance2Score(distance, habbits=64, num=3):
    yuzhi = round((float(habbits - num) / habbits) * 100, 2)
    isSim = ""
    if distance > 64:
        score = round(float(0), 2)
    else:
        score = round((float(habbits - distance) / habbits) * 100, 2)
    if score >= yuzhi:
        isSim = "YES"
    else:
        isSim = "NO"
    return yuzhi, score, isSim


def sqlConnect(host="118.89.139.154", port=20007, user="root", passwd="somao1129", db="tianyancha", charset="utf8"):
    """
    connect the mysql
    """
    mysqlConnect = dict(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    try:
        conn = MySQLdb.connect(**mysqlConnect)
    except Exception as e:
        print e
    else:
        return conn

def compSimHashValue(companyBusiness,idf_filter={},stop_words=[]):
    """

    :param companyBusiness: str format describe the product info
    :return:
    """
    token = tokenize(companyBusiness,stop_words)
    for word in token:
        if not idf_filter.has_key(word):
            token.remove(word)
    if not token:
        print "no word supply"
        return {}
    new_sentence = "".join(token)
    print "new sentence: ",new_sentence
    tags = keywordExtractWithWeight(new_sentence, typeExtract='tag_extract', K=2)
    tags_ls = ['|'.join((word, str(weight))) for (word, weight) in tags]
    tag_feature = ", ".join(tags_ls)
    print "tag_feature:", tag_feature
    if not tag_feature:
        return ([])
    mysimhash = MySimhashComp(tags, 2, 64)
    comp_simhash = mysimhash.simhash
    return comp_simhash

def is_same_company(comp,corp):
    """
    compare two company is relation with sub company
    :param comp: company1 string
    :param corp: company2 string
    :return: boolean
    """
    comp_seg = pseg.cut(comp)
    corp_seg = pseg.cut(corp)
    wordcomp_name = {}
    wordcorp_name = {}
    for word,pos in comp_seg:
        print "word={} and pos={}".format(word.encode('utf-8'),pos)
        if pos == 'nz':
            wordcomp_name['nz']=word
            break
        elif pos == 'ns':
            wordcomp_name['ns']=word
            break
    for word,pos in corp_seg:
        print "word={} and pos={}".format(word.encode('utf-8'), pos)
        if pos == 'nz':
            wordcorp_name["nz"]=word
            break
        elif pos == 'ns':
            wordcorp_name['ns']=word
            break
    if wordcorp_name.has_key('nz') and wordcomp_name.has_key('nz'):
        if wordcorp_name['nz']==wordcomp_name['nz']:
            return True
    elif wordcorp_name.has_key('ns') and wordcomp_name.has_key('ns'):
        comp_sub = comp.replace(wordcomp_name['ns'],'')
        print "comp_sub",comp_sub
        if not type(comp_sub) == unicode:
            comp_sub = comp_sub.decode('utf')
        corp_sub = corp.replace(wordcorp_name['ns'], '')
        print "corp_sub",corp_sub
        if not type(corp_sub) == unicode:
            corp_sub = corp_sub.decode('utf')
        if comp_sub[:2] == corp_sub[:2]:
            return True
    return False

def fileCount(path):
    """
    get all the file name in the path and return a count
    :param path: the path to be retrieve file
    :return: count for files int
    """
    count = 0
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(file):
            count += 1
    return count
def odbc_connect(**kwargs):
    """
    kwargs info:host,port,database,user,passwd
    """
    odbcHandler = 'DRIVER={SQL Server};SERVER=%s,%s;DATABASE=%s;UID=%s;PWD=%s' %(kwargs["host"],kwargs["port"],kwargs['db'],kwargs["user"],kwargs["passwd"])
    try:
        conn = pyodbc.connect(odbcHandler)
    except Exception as e:
        print e
    else:
        return conn

path = "D:\\jwh\\nlp\\stop_words_new.txt"
set_stop_words(path)
def textClean(text):
    """
    remove non-sense symbols in operation range text
    :param text: operation range text with string format
    """
    text = re.subn('<[^>]*>','',text)[0]
    text = re.subn('\s+','',text)[0]
    text = re.subn('\d+','',text)[0]
    text = re.subn('\(([\w\W]*?)\)','',text)[0]
    text = re.subn('\[([\w\W]*?)\]','',text)[0]
    text = re.subn(u'（([\w\W]*?)）','',text)[0]
    text = re.subn(u'【([\w\W]*?)】','',text)[0]
    text = re.subn(u'『([\w\W]*?)』','',text)[0]
    #text = re.sub("(&nbsp;)+","",text)
    #text = re.sub('&ensp;','',text)
    text = re.sub('&[A-Za-z]+;','',text)
    text = re.sub('[\+\.\=<>\?]+','',text)
    return(text)

def clean_stop_word(words,path):
    stop_words = []
    with open(path,'rb') as f:
        for line in f.readlines():
            line = re.sub('\s+','',line.strip())
            stop_words.append(line)
    for word in words:
        if word in stop_words:
            words.remove(word)
    return words
def keywordExtractWithWeight(sentence,typeExtract="textRank",K=20):
    sentence = textClean(sentence)
    type_list = ["tag_extract","textRank","snow","pynlp"]
    tags = []
    if typeExtract==type_list[0]:
        try:
            #print "extract from jieba.extract_tags"
            tags = extract_tags(sentence,withWeight=True,topK=K)
        except Exception as e:
            print e.message
    if typeExtract==type_list[1]:
        try:
            #print "extract from jieba.textrank"
            tags = textrank(sentence,withWeight=True,topK=K)
        except Exception as e:
            print e.message
    if typeExtract==type_list[2]:
        try:
            print "extract from snowNLP"
            snow = SnowNLP(sentence)
            tagsAll = snow.tags
            keywords = snow.keywords(limit=K)
            print "|".join(keywords)
            tags = []
            for i,(word,tag) in enumerate(tagsAll):
                print "word info:(%s,%s)" %(word.encode("utf-8"),tag)
                if word in keywords:
                    tags.append(tagsAll[i])

        except Exception as e:
            print e.message
    if typeExtract == type_list[3]:
        try:
            print "extract from pynlp"
            tags = pynlpir.get_key_words(sentence,max_words=K,weighted=True)
        except Exception as e:
            print traceback.format_exc(1)
            print e.message
    return tags


def keywordExtract(sentence):
    keyword_dict = {}
    cut_words = list(jieba.cut(sentence,cut_all=False))

    cut_words = clean_stop_word(cut_words,path)
    for word in cut_words:
        print "cut result: ",word
    segWords = ""
    segList = pseg.cut(str(cut_words))
    for seg in segList:
        if seg.flag in ['n','vn'] and len(seg.word) > 1:
            segWords += seg.word
            print "seg result: %s" %segWords
    try:
        jieba_keywords = extract_tags(segWords,topK=10,allowPOS=('nv','n'))
        for key in jieba_keywords:
            print "jieba extract info:", key
    except:
        print traceback.format_exc(1)
        jieba_keywords = []
    try:
        text_keywords = textrank(segWords,topK=10,allowPOS=('nv','n'))
        for key in text_keywords:
            print "textrank info:", key
    except:
        print traceback.format_exc(1)
        text_keywords = []
    try:
        snowTag = SnowNLP(segWords)
        snow_keywords = snowTag.keywords(limit = 10)
        for key in snow_keywords:
            print "snowNlp result:",key
    except:
        print traceback.format_exc(1)
        snow_keywords = []
    try:
        pynlpir_keywords = pynlpir.get_key_words(segWords,max_words=10)
        for key in snow_keywords:
            print "pynlp result:",key
    except:
        print traceback.format_exc(1)
        pynlpir_keywords = []
    try:
        vectorCount = CountVectorizer()
        transform = TfidfTransformer()

        vectorCount.fit_transform(cut_words)
        sklearn_keywords = vectorCount.get_feature_names()
        for key in sklearn_keywords:
            print "sklearn result:", key
    except:
        print traceback.format_exc(1)
        sklearn_keywords = []
    try:
        keywords_combine = jieba_keywords + text_keywords + snow_keywords + pynlpir_keywords + sklearn_keywords
        print "combine successfully"
    except Exception as e:
        print "type not match"
    else:

        for keyword in keywords_combine:
            if not keyword_dict.has_key(keyword):
                keyword_dict[keyword] = 0
            keyword_dict[keyword] += 1.0/5
    keyword_dict_cp = keyword_dict.copy()
    for key in keyword_dict:
        if keyword_dict[key] < 0.6:
            keyword_dict_cp.pop(key)
    return keyword_dict_cp.keys()


class RandomVec(object):
    def __init__(self,vector_size):
        self.new_vec = []
        self.random_vec = {}
        self.size = vector_size

    def __getitem__(self, item):
        get_id = self.random_vec.get(item,-1)
        if get_id == -1:
            new_vec = np.array([random() for i in range(self.size)])
            self.random_vec[item] = len(self.new_vec)
            self.new_vec.append(new_vec)
            return new_vec
        else:
            return self.new_vec[get_id]

emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude00-\udeff])|"  # transport & map symbols
    u"(\ud83c[\u0000-\uffff])|"  # undefined
    u"(\ud852[\u0000-\uffff])|"  # undefined
    u"(\ud83d[\u0000-\uffff])|"  # undefined
    u"(\ud83e[\u0000-\uffff])|"  # undefined
    u"([\ud800-\udfff][\u0000-\uffff])|"  # undefined
    u"(\udfb4[\u0000-\uffff])|"  # really rare chars can't decode
    u"(\ud83c[\udd00-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)

def remove_emoji(text):
    if not isinstance(text,unicode):
        text = text.decode("utf")
    return emoji_pattern.sub(ur'', text)

def clean(text):
    if not type(text) == unicode:
        print "change to unicode for clean..."
        text = text.decode("utf-8")
    # text = full_to_half(text)
    # text = re.sub(u"^[^\u4E00-\u9FA5]+","",text)
    # text = re.sub(u"\{[^\u4E00-\u9FA5]*\}","",text)
    # text = re.sub(u"\[[^\u4E00-\u9FA5]*\]","",text)
    # text = re.sub(u"\([^\u4E00-\u9FA5]{1,2}","",text)
    # text = re.sub(u"[^\u4E00-\u9FA5]{1,2}\)","",text)
    # text = re.sub(u"[^\u4E00-\u9FA5]+$","",text)
    text = re.sub(u"[^\w\d\u4E00-\u9FA5]+$", "", text)
    text = re.sub(u"\s+","",text)
    text = re.sub(u"\(.*?\)","",text)
    text = re.sub(u"\（.*?\）","",text)
    text = re.sub(u"\[.*?\]", "", text)
    text = re.sub(u"\{.*?\}", "", text)
    text = re.sub(u"<.*?>", "", text)
    text = re.sub(u"<<.*?>>", "", text)
    text = re.sub(u"【.*】","",text)
    # text = half_to_full(text)
    text = re.sub(u"[^\w\d\u4E00-\u9FA5。？！，：\.\?\!,:]","",text)
    pattern_rep = re.compile(ur"([^^\w\d\u4E00-\u9FA5])(\1+)")
    text = re.sub(pattern_rep,ur'\1',text)
    return text

def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        #0x3000表示中文的全角空格符,是unicode的16进制表示(0x开头,8进制一般以0开头),对应的十进制是12288,32是英文半角空格的ascii(0-255),也是十进制的unicode值,通过chr还原成字符
        if num == 0x3000:
            num = 32
        #0xFF01,0xFF5E分别是中文字符信息'！','～'
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = unichr(num)  #python3 直接使用chr
        n.append(char)
    return ''.join(n)

def half_to_full(s):
    """
    convert half-width charater to ful
    :param s:
    :return:
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 32:
            num = 0x3000
        elif 0x21 < num < 0x7E:
            num += 0xfee0
        char = unichr(num)
        n.append(char)
    return ''.join(n)


def get_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formater = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh.setFormatter(formater)
    fh.setFormatter(formater)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def constellation_compute(time_str):
    """
    compute the contellation from date
    :param time_str: input string date with format 'yyyy-mm-dd'
    :return:
    """
    if not time_str:
        return
    pat = re.compile(r'\d{2,4}[^\d]+\d{1,2}[^\d]+\d{1,2}')
    mat_pat = re.match(pat,time_str)
    if not mat_pat:
        return
    digits = re.findall(r'\d+',time_str)
    time_str = "-".join(digits)
    try:
        date_strp= datetime.strptime(time_str,'%Y-%m-%d')
    except:
        date_strp = datetime.strptime(time_str, '%y-%m-%d')
    mon = date_strp.month;day = date_strp.day
    if mon ==1:
        if day <=19:
            return u"摩羯座"
        else:
            return u"水瓶座"
    elif mon == 2:
        if day <= 18:
            return u"水瓶座"
        else:
            return u"双鱼座"
    elif mon == 3:
        if day <=20:
            return u"双鱼座"
        else:
            return u"白羊座"
    elif mon == 4:
        if day<=19:
            return u"白羊座"
        else:
            return u"金牛座"
    elif mon == 5:
        if day<=20:
            return u"金牛座"
        else:
            return u"双子座"
    elif mon == 6:
        if day <= 21:
            return u"双子座"
        else:
            return u"巨蟹座"
    elif mon == 7:
        if day <=22:
            return u"巨蟹座"
        else:
            return u"狮子座"
    elif mon == 8:
        if day <= 22:
            return u"狮子座"
        else:
            return u"处女座"
    elif mon ==9:
        if day <= 22:
            return u"处女座"
        else:
            return u"天秤座"
    elif mon == 10:
        if day <= 23:
            return u"天秤座"
        else:
            return u"天蝎座"
    elif mon == 11:
        if day <= 22:
            return u"天蝎座"
        else:
            return u"射手座"
    else:
        if day <= 21:
            return u"射手座"
        else:
            return u"摩羯座"


def age_compute(birth_day):
    """
    compute age
    :param birth_day: string format: "yyyy-MM-dd"
    :return: int age scale 0-100
    """
    year_now = datetime.now().year
    month_now = datetime.now().month
    if not birth_day:
        return
    pat = re.compile(r'\d{2,4}[^\d]+\d{1,2}[^\d]+\d{1,2}')
    mat_pat = re.match(pat,birth_day)
    if mat_pat:
        digits = re.findall(r'\d+',birth_day)
        birth_day = "-".join(digits)
        try:
            birth_date = datetime.strptime(birth_day,"%Y-%m-%d")
            birth_year = birth_date.year
        except:
            birth_date = datetime.strptime(birth_day,"%y-%m-%d")
            birth_year = birth_date.year
        age = year_now - birth_year
        if age <=0:
            age = 0
        elif age>=100:
            age = 100
        return age
    return

# className don't be the same with fileName at best
class MySimhashComp:
    def __init__(self, tokens, kaggle, habbits):
        self.tokens = tokens
        self.kaggle = kaggle
        self.habbits = habbits
        self.simhash = self.simhash()

    def __str__(self):
        return str(self.hash)

    def simhash(self):
        """
            @description: compute the simhash value with the result of tokenize
            @params:
            @params:
            @params:
            @return: string: simhash value
        """
        hash_arr = []
        for feature, weight in self.tokens:
            # print "权重信息",weight
            weight = int(self.kaggle * weight)
            hash = self.strhash(feature)
            if len(hash) < 64:
                print "no hash value"
                continue
            try:
                new_hash = [None for x in range(len(hash))]
                for i in range(len(hash)):
                    if hash[i] == "1":
                        new_hash[i] = weight
                    else:
                        new_hash[i] = -weight
            except Exception as e:
                print e.message
            hash_arr.append(new_hash)
        # print hash_arr
        sum_hash = list(np.array(hash_arr).sum(axis=0))
        # print "特征合并后的hash结果："
        # print sum_hash
        if not sum_hash:
            return "00"
        simhashValue = ""
        for x in sum_hash:
            if x > 0:
                simhashValue += '1'
            else:
                simhashValue += '0'
        # print "simhash值：%s" %simhashValue
        return simhashValue

    def strhash(self, source):

        if not type(source) == unicode:
            source = source.decode('utf-8')

        if source == "":
            return "00"
        else:
            x = ord(source[0]) << 7
            mask = 1000003
            m = 2 ** self.habbits - 1
            for c in source:
                # print "unicode字节c的16进制形式:%s" % c.encode('utf-8')
                x = ((x * mask) ^ ord(c)) & m
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace("0b", "").zfill(self.habbits)[-self.habbits:]
            # print "原生输入和当前转化后的结果：(%s,%s)" % (source.encode("utf-8"), x)
            return str(x)

    def haimingDis(self, other):
        """
            @description: compute the haiming distance between two simhash value
            @params:    two simhash value between this and that
            @params:
            @params:
            @return:    int:haimingdistance
        """
        """
        this = self.simhash
        that = other.simhash
        distance = 0
        for i in range(len(this)):
            if this[i] != that[i]:
                distance += 1
            else:
                distance += 0
        return distance
        """
        this = "0b" + self.simhash
        that = "0b" + other.simhash
        # print "对比两片文档simhash的结果："
        # print this
        # print that
        n = int(this,2) ^ int(that,2) #转化为10进制进行异或计算
        # print "二进制取异或后的结果：%s" %n
        distance = 0
        while n:
            n &= (n-1)
            distance += 1
        return distance

    def distance2Score(self,distance,num):
        """
            @description: compute the score according to the distance and
                        reference the yuzhi define the is similarity , yuzhi: distance=3,score=61/64*100
            @params:  distance from the haimingDistance
            @params:
            @params:
            @return: yuzhi,score, similarity
        """
        yuzhi = round((float(self.habbits - num)/self.habbits)*100,2)
        isSim = ""
        if distance>64:
            score = round(float(0),2)
        else:
            score = round((float(self.habbits - distance)/self.habbits)*100,2)
        if score >= yuzhi:
            isSim = "YES"
        else:
            isSim = "NO"
        return yuzhi,score,isSim

import string
def extract_proc(input_sentence):
    zhuyu_flag = False;
    symbols = string.punctuation;
    tmp_proc = []
    count = 0;
    symbols_pos = 0;
    weiyu_flag = False
    tokens = psg.cut(input_sentence, HMM=True)
    for word, flag in tokens:
        count += 1
        if flag in ('n', 'nr', 'r', 'nz', 'nt', 'ns'):
            zhuyu_flag = True
            zhuyu_pos = count
            tmp_zhuyu = word
        if word in symbols:
            symbols_pos = count
        if zhuyu_flag and (symbols_pos < zhuyu_pos):
            if flag in ("v", "vn"):
                weiyu_pos = count
                tmp_weiyu = word
                weiyu_flag = True
                tmp_proc = [tmp_zhuyu, tmp_weiyu]
                zhuyu_flag = False
        if word in symbols:
            symbols_pos = count
        if weiyu_flag and (symbols_pos < weiyu_pos):
            if flag in ("n", 'nr', 'r', 'nz', 'nt', 'ns'):
                tmp_proc.append(word)
                break
    return tmp_proc

def suit_sentence(sentence,limit_chars):
    if not type(sentence) != unicode:
        sentence = sentence.decode("utf")
    proc = extract_proc(sentence)
    if proc and len(sentence) < limit_chars:
        return True
    return False

def save_pickle(file_obj,file_path):
    with open(file_path,'wb') as fw:
        pickle.dump(file_obj,fw)

def load_pickle(file_path):
    with open(file_path,"rb") as fr:
        pickle_data = pickle.load(fr)
    return pickle_data















