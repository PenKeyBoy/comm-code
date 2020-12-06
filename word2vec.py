#!/usr/bin/env/py35
# coding=utf-8
from __future__ import print_function
from gensim.models.word2vec import Word2Vec
import codecs
import numpy as np
from random import random
import argparse
import sys
import pdb
import os
import re

def load_sentences(data):
    count = 0
    sentences = []
    sentence = []
    for line in data:
        count += 1
        if count > 1:
            if line.startswith('-DOC'):
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line.rstrip('\r\n'))

    sentences.append(sentence)
    return sentences

def signal_sentences_specify(path,zeros=False):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = re.sub('\d','0',line.rstrip()) if zeros else line.rstrip()   #rstrip()去除字符串右边的换行符和空格
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

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

class WordVec(object):
    def __init__(self,args):
        self.sentences = []
        self.sentence = []
        self.vocab = set()
        if args.restore is None:
            with codecs.open(args.file,'r',encoding='utf-8') as fr:
                for count,line in enumerate(fr):
                    if count > 1 and line.startswith('-DOC'):
                        self.sentences.append(self.sentence)
                        self.sentence = []
                    else:
                        try:
                            word,tag = line.rstrip('\r\n').split()
                        except Exception as e:
                            print("no enough element to unpack line {} is: {} with error:{}".format(count+1,line,e))
                        else:
                            for char in word:
                                self.sentence.append(char)
                                self.vocab.add(char)
            #pdb.set_trace()
            self.wordvec = Word2Vec(sentences=self.sentences,size=args.vector_size,window=args.window,min_count=args.min_count,max_vocab_size=len(self.vocab),
                               workers=args.workers,sg=args.sg,batch_words=args.batch_size)

        else:
            self.wordvec = Word2Vec.load_word2vec_format(args.restore)
        self.randvec = RandomVec(args.vector_size)

    def __getitem__(self,word):
        try:
            vector = self.wordvec[word]
            return vector
        except:
            vector = self.randvec[word]
            return vector

    def vector2file(self,args):
        #lines = []
        with codecs.open(args.file2write,'w',encoding='utf8') as fw:
            fw.write("总的字个数:" + str(len(self.vocab)) + ' ' + "向量长度：" + str(args.vector_size) + '\n')
            for count,word in enumerate(self.vocab):
                vector2list = [str(value) for value in self[word]]
                line = [word] + vector2list
                assert len(line)==args.vector_size + 1
                fw.write(' '.join(line))
                if count < len(self.vocab)-1:
                    fw.write('\n')
        print("sucessfully generate vector_file")


if __name__ == '__main__':
    script_name = os.path.basename(sys.argv[0])
    print(script_name)
    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(usage="%s %s [options]" %(sys.executable,script_name),description="word2vector commandline")
    parser.add_argument('-f','--file',type=str,help="path to wait for training",required=True)
    parser.add_argument('-s','--vector_size',type=int,help='specify the word_size',required=True)
    parser.add_argument('-fw','--file2write',type=str,help="destination file to write",required=True)
    parser.add_argument('-w','--window',type=int,help="specify the window to select",default=5)
    parser.add_argument('-mc','--min_count',type=int,help="specify the min count to compute",default=5)
    parser.add_argument('--workers',type=int,help="specify the workers number to computer for parallel",default=3)
    parser.add_argument('--sg',type=int,help="choose the method to train sample skip_gram if 0 else 1 represent cbow",default=0)
    parser.add_argument('-b','--batch_size',type=int,help="choose the numbers for batch to specify the batch size",default=1000)
    parser.add_argument('-r','--restore',type=str,help="path to store the pre_train vector model",default=None)
    args = parser.parse_args()
    wordvec = WordVec(args)
    wordvec.vector2file(args)
