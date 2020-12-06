#!/usr/bin/env/py35
# -*-coding:utf-8-*-
# !/usr/bin/python3
# -*- coding:utf-8 -*-
import re
from numpy import log
import operator
#from pandas import DataFrame
import pymysql
import time
import jieba
import jieba.posseg as pseg
import logging
from jieba.analyse import set_stop_words
# from word2vecSIm import stopWord,tokenize
from datetime import datetime
import pickle
#global path,stop_words
#path = "/home/jwh/nlp/stop_words_new.txt"

#set_stop_words(path)
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
logger = get_logger("construct_word.log")

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



if __name__ == '__main__':

    # according the score sort the result
    #summary_title = False

    insert2db = True
    t0 = time.time()
    mysqlDict = dict(host="118.89.139.154", port=20007, user="root", passwd="somao1129", db="tianyancha", charset="utf8")
    try:
        connect = pymysql.connect(**mysqlDict)
    except:
        print("can't connect the mysql")
    else:
        cursor = connect.cursor()
        cursor.execute("select business from tianyancha_product_clean_tmp")

        news_pairs = cursor.fetchall()

        text = ""
        for pair in news_pairs:
            business_conn = ","+pair[0]
            text += business_conn
            #print(text)
            #print('='*20)
        # stop_words = stopWord(path)
        t1 = time.time()
        """
        score = garm_entroy(text)    #0.89min,total=1.39min
        print("新词构造的得分结果： ",score)
        score_sort = sorted(score.items(),key=lambda x:x[1][2],reverse=True)
        t2 = time.time()
        delta1 = (t2 - t1)/60
        print("pmi计算耗时：%.2f(min)" % delta1)
        """
        #计算概率比值的结果
        degree = combineDegree(text)
        sort_degree = sorted(degree.items(),key=operator.itemgetter(1),reverse=True)
        t3 = time.time()
        delta2 = (t3 - t1)/60
        print("概率比较计算耗时：%.2f(min)" % delta2)  #cost:35min
        """
        for construct_word, score_info in score_sort[:20]:
            print("创建新词的信息：(%s: %s)" %(construct_word,score_info))

            #construct_word_ls.append(construct_word)
            #score_ls.append(score_info)
        """
        print("\n直接计算概率值，设置阈值是20，输出频率前二十的条目: ", sort_degree[:20])

        if insert2db:
            f = open('/usr/nlp/construct_word_prob.txt', mode='w', encoding='utf-8')
            '''
            for construct_word,score_info in score_sort:
                cursor.execute("""insert into tianyancha_product_business_construct_word_by_entroy(construct_word,score,freq,scoreMultiFreq)
                                values ('%s','%s','%s','%s')
                                """ % (construct_word,score_info[0],score_info[1],score_info[2]))
                connect.commit()
                #print("创建新词的信息：(%s: %s)" %(construct_word,score_info))
                f.write(construct_word+'\n')
                #construct_word_ls.append(construct_word)
                #score_ls.append(score_info)

            '''

            for word,freq in sort_degree:
                #cursor.execute("insert into tianyancha_product_business_construct_word_by_prob(word,freq) values('%s','%s')" % (word,freq))
                #connect.commit()
                f.write(word+'\n')

            f.close()
        cursor.close()
        connect.close()
        t4 = time.time()
        #delta3 = (t4 - t2)/60
        #print("信息熵插入数据耗时：%.2f(min)" % delta3) # cost:0.46min
        delta4 = (t4 - t0)/60
        print("总共耗时：%.2f(min)" % delta4)

    '''
    
    construct_word_ls_new = []
    score_ls_new = []
    for construct_word,score_info in new_score_record[:20]:
        print("比较后,剔除重复的信息：(%s: %s)" %(construct_word,score_info))
        construct_word_ls_new.append(construct_word)
        score_ls_new.append(score_info
    score_df = DataFrame(score_ls_new,index=construct_word_ls_new,columns=["信息熵得分","词频","总得分"])
    score_df.head()
    #分段读入
    with open("/home/jwh/nlp/construct_word/score_withoutStopWords_" + title[:4] + ".txt", 'w') as f:
       for ele in construct_word_ls_new:
           f.write('\n')
           f.writelines(ele)
    
    #score_df.to_csv("~/nlp/construct_word/construct_score"+title[:4]+".csv")
    #test sentence
    test_tokens = ["智能","客服","系统","问答","智能","客服","机器人","语音","技术"]

    scores = garm_entroy(test_tokens)
    print("分词后的结果集进入模型训练： ",scores)

    score_sort = sorted(list(scores.items()), key=lambda x: x[1][2], reverse=True)
    select_num = 20
    print(score_sort[:5])
    dict_score = {}
    construct_word_ls = []
    score_ls = []
    count = 0
    for construct_word, score_info in score_sort[:20]:
        count += 1
        print("创建第%d新词的信息：(%s: %s)" % (count, construct_word, score_info))
        construct_word_ls.append(construct_word)
        score_ls.append(score_info)
    new_score_record = compare_score(score_sort[:20])
    print("筛选过后的结果： ",new_score_record)

    degree = combineDegree(test_tokens)
    print("直接计算概率值，设置阈值是10: ",degree)
    '''








