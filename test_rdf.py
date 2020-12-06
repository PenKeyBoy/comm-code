# !/usr/bin/python
# coding=utf-8

from urlparse import urljoin
from rdflib import Graph,URIRef
import codecs
import pickle
import sys
reload(sys)
if sys.getdefaultencoding()!= "utf-8":
    sys.setdefaultencoding("utf-8")
g = Graph()

baike_base_url = "https://baike.baidu.com/item"
lemma2id_fpath = "../lemma2id_dic.pkl"
rdf_file = "baike.rdf"
# 解析rdf文件
g.parse(rdf_file, format="xml")
lemma2id = pickle.load(codecs.open(lemma2id_fpath,'rb'))
def test(subject=None, predicate=None, object=None):

    if not predicate:
        print "parse predicate"
        if subject and object:
            if lemma2id.has_key(subject):
                subjects = lemma2id[subject]
                for idx in range(len(subjects)):
                    preds = g.predicates(URIRef(urljoin(baike_base_url, subjects[idx])),
                                     URIRef(urljoin(baike_base_url, object)))
                    for pred in preds:
                        pred = pred.split("/")[-1]
                        print "%s是%s的%s" % (object, subject, pred)
                        break
                    break
            else:
                print "subject %s not in lemma dict" %subject
    elif not object:
        print "parse object"
        if subject and predicate:
            if lemma2id.has_key(subject):
                subjects = lemma2id[subject]
                for idx in range(len(subjects)):
                    objects = g.objects(URIRef(urljoin(baike_base_url, subjects[idx])),
                                        URIRef(urljoin(baike_base_url, predicate)))
                    for obj in objects:
                        obj = obj.split("/")[-1]
                        print "%s的%s是%s" % (subject, predicate, obj)
                        break
                    break
            else:
                print "subject %s not in lemma dict" % subject
    else:
        pass


if __name__ == '__main__':
    #is_what句式的匹配

    try:
        # 宋晓锋和小沈阳是什么关系
        test(subject="宋晓峰",object="小沈阳")
        #宋晓锋的搭档是谁
        test(subject="宋晓峰",predicate="搭档")
    except Exception as e:
        print e
    if g:
        g.close()