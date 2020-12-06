# -*- coding: utf-8 -*-

from flask_bootstrap import Bootstrap
from flask import Flask,render_template,session,request,jsonify
from flask_classy import FlaskView,route
import sys,os
import codecs
import time
import traceback
import redis
import requests
import json
import random
from rdflib.graph import Graph
from rdflib.term import BNode,URIRef,Literal

from urlparse import urljoin
from common_utils import get_logger,remove_emoji
from match import Match,main
from src_file import SrcFiles
from qa_conclude_info import ans_libs,emoji_lib
# from qa_conclude_info import qa_host_info,qa_chat_info,self_label
# from qa_conclude_info import host_info, chatbot_info, user_info
import threading
from mpe_parse import MPEparse,jieba_add_word,mpe_add_rule
from nostanford_rule import NoStanfordRule


mpeParse = MPEparse()
jieba_add_word()
# mpe_add_rule(mpeParse)


cur_dir = os.getcwd()
if cur_dir.endswith("src"):
    cur_dir = os.path.dirname(cur_dir)
logger = get_logger(os.path.join(cur_dir, "log", "match_for_api.log"))
logger.info("current dir:%s" % cur_dir)

if cur_dir.endswith("chatbot"):
    cur_dir = os.path.join(cur_dir,"src")

app=Flask(__name__,template_folder=os.path.join(cur_dir,"template"))
app.secret_key='\xf1\x92Y\xdf\x8ejY\x04\x96\xb4V\x88\xfb\xfc\xb5\x18F\xa3\xee\xb9\xb9t\x01\xf0\x96'    #配置secret_key,否则不能实现session对话
bootstrap=Bootstrap(app)#Flask扩展一般都在创建实例时初始化，这行代码是Flask-Bootstrap的初始化方法

srcFiles = SrcFiles("convers_corpus")
proc_trie, idf_trie, answer_indexs = Match.load_file(srcFiles.store_path)
proc_ratio = Match.proc_ratio(0.2, 0.5)

src_trie = srcFiles.get_src_trie
g = Graph()
BASE_URL = "http://0.0.0.0:5002"



class RedisCache():
    def __init__(self,client):
        self.client = client

    def set_user(self,input_timestamp,user_input):
        if not user_input:
            user_input = ""
        self.client.hset("user_info",str(input_timestamp),user_input)

    def get_user(self,input_timestamp):
        user_info = self.client.hget("user_info",input_timestamp)
        if not user_info:
            return
        return user_info

    def set_chatbot(self,label,value):
        if not value:
            value = ""
        self.client.hset("chatbot_info",str(label),value)
    def get_chatbot(self,label):
        chatbot_info = self.client.hget("chatbot_info",label)
        if not chatbot_info:
            return
        return chatbot_info
    def set_host(self,label,value):
        if not value:
            value = ""
        self.client.hset("host_info",str(label),value)
    def get_host(self,label):
        user_info = self.client.hget("host_info",label)
        if not user_info:
            return
        return user_info
    def del_info(self):
        if self.client.exists("host_info"):
            self.client.delete("host_info")
        if self.client.exists("chatbot_info"):
            self.client.delete("chatbot_info")
        if self.client.exists("user_info"):
            self.client.delete("user_info")

cache = RedisCache(redis.StrictRedis())


@app.route('/', methods=['GET','POST'])
@app.route('/index', methods=['GET','POST'])
def index():
    if session.get('qa')!=None:
        if session.get('ip')!=None:
            with codecs.open(os.path.join(os.path.dirname(cur_dir),"log",session['ip']),"ab+",'utf8') as fw:
                for pair in session.get('qa'):
                    try:
                        fw.write('%s|$|%s\n' % (pair[0],pair[1]))
                    except Exception as e:
                        print e
        session.pop('qa')
    return render_template('base.html')

@app.route('/chat', methods=['GET','POST'])
def chat():
    base_label = ["name", "old", "gender", "constellation"]
    user_info = {};
    host_info = {"userAge":18,"sex":1,"hobby":u"吃鸡睡觉打豆豆"};
    chatbot_info = {"name":u"小可爱","sex":u"纯爷们儿","age":2}
    # qa_info = {"host": qa_host_info, "user": self_label, "chatbot": qa_chat_info}
    if session.get('qa')==None:
        session['qa'] = []
    if request.method=='POST':
        if session.get('ip')==None:
            session['ip'] = request.remote_addr.replace(".","_")
        sentence = request.form['sentence']
        session_id = str(random.random()).split(".")[-1]
        answer_type = ""
        t0 = time.time()
        try:
            rule = NoStanfordRule(qa_host_label=host_info,qa_chat_label=chatbot_info)
            match = Match(sentence, logger,session_id,rule)
            # if match.user_info:
            #     for _key in match.user_info:
            #         cache.set_user(_key,match.user_info[_key])
            # for _base in base_label:
            #     cache_user = cache.get_user(_base)
            #     cache_host = cache.get_host(_base)
            #     cache_chatbot = cache.get_chatbot(_base)
            #     if cache_user:
            #         user_info.update({_base: cache_user})
            #     if cache_host:
            #         host_info.update({_base: cache_host})
            #     if cache_chatbot:
            #         chatbot_info.update({_base: cache_chatbot})

            # for _k in user_info:
            #     print("user key= %s and value= %s" %(_k,user_info[_k]))
            # object_info = {"host": host_info, "user": user_info, "chatbot": chatbot_info}
            answer, answer_type, answer_ind, _, _ = main(match, proc_trie, proc_ratio,
                                                         idf_trie,
                                                         answer_indexs, srcFiles.template_file)
            if answer_ind:
                try:
                    from_ques = src_trie[answer_ind]
                except:
                    pass
            code = 10000
            del match
        except:
            logger.info(traceback.format_exc())
            answer = random.choice(ans_libs)
            code = 20000
        t1 = time.time()
        t_delta = t1 - t0
        # response = jsonify({"code":code,"response":answer,"match_type":answer_type,"time_response":t_delta})
        if isinstance(answer, unicode):
            answer = answer.encode("utf-8")
        # res = {"code": code, "answer": answer, "from_ques": from_ques, "match_type": answer_type, "time_response": t_delta}
        _answer = [answer,answer_type,t_delta,code]
        try:
            print "answer:\t\n%s\n" % _answer
        except Exception:
            pass
        session['qa'] = [[sentence]+_answer]+session['qa']
        try:
            print session
        except Exception:
            pass        
        return render_template('chat.html',qa=session.get('qa'),qa_len=len(session.get('qa')))
    return render_template('chat.html',qa=session['qa'],qa_len=len(session['qa']))


@app.route('/answer')
def answer(): #根据chainId,查询aes_key
    with codecs.open(os.path.join(cur_dir,"answer.dat"),'r',"utf8") as fw:
        answer_dat = "<br>".join([line.strip() for line in fw.readlines()])
    return render_template('answer.html',answer_dat=answer_dat)

@app.route('/data_backend',methods=["GET","POST"])
def data_backend():
    cache.del_info()
    if request.method == "POST":
        user_id = request.form.get("userId")
        pet_id = request.form.get("petId")
        chatbot_info = srcFiles._get_chatbot_info(pet_id, "pet")
        if chatbot_info:
            for _key in chatbot_info:
                if type(chatbot_info[_key]) in (str,int,float,unicode):
                    cache.set_chatbot(_key,chatbot_info[_key])
        user_info = srcFiles._get_user_info(user_id, "user")
        if user_info:
            for _key in user_info:
                if type(user_info[_key]) in (str, int, float, unicode):
                    cache.set_user(_key,user_info[_key])
        try:
            host_info = srcFiles._get_user_info(chatbot_info["user_id"], "user")
            for _key in host_info:
                if type(host_info[_key]) in (str, int, float, unicode):
                    cache.set_host(_key,host_info[_key])
        except:
            pass
        # return jsonify({"host_info":cache.get_info("host"),"user_info":cache.get_info("user"),"chatbot_info":cache.get_info("chatbot")})
        # return render_template("data.html",cache="已缓存")
        return jsonify({"cache":cache})
    # return jsonify({"host_info":"","user_info":"","chatbot_info":""})
    # return render_template("data.html",cache="清空缓存")
    return jsonify({"cache":cache})

class ChatterBotDouban(FlaskView):

    def __init__(self):
        self.cn = 0
        # self.g = Graph()
    @route('/chatterbot/<input_sentence>')
    def get_dir_response(self,input_sentence):
        t0 = time.time()
        user_info = srcFiles._get_user_info("10", "user")
        print("user_id=10 info:\n",user_info)
        chatbot_info = srcFiles._get_chatbot_info("7b7924503aaf48b5b45bb1a30c69f14c","pet")
        print("chatbot info:\n",chatbot_info)
        try:
            host_info = srcFiles._get_user_info(chatbot_info["user_id"],"user")
        except:
            host_info = {}
        print("user_id={} info:{}\n".format(chatbot_info["user_id"],host_info))
        # object_info = {"host": host_info, "user": user_info, "chatbot": chatbot_info}
        # qa_info = {"host": qa_host_info, "user": self_label, "chatbot": qa_chat_info}
        from_ques = u""
        try:
            match = Match(input_sentence, logger)
            answer,answer_type,answer_ind,_,_ = main(match,proc_trie,proc_ratio,idf_trie,answer_indexs,srcFiles.template_file)
            status = 0
            if answer_ind:
                try:
                    from_ques = src_trie[answer_ind]
                except:
                    pass
            del match
        except:
            print(traceback.format_exc())
            answer = "error ocurr in match"
            status = 1
        t1 = time.time()
        t_delta = t1 - t0
        response = jsonify({"response":answer,"match_type":answer_type,"match_ques":from_ques,"status":status,"time_response":t_delta})
        return response

    @route('/chatterbot',methods=['POST'])
    def get_simple_response(self):
        params = request.get_json()
        t0 = time.time()
        answer_type = ""
        base_label = ["name","old","gender","constellation"]
        user_info = {};host_info={};chatbot_info={}
        # qa_info = {"host": qa_host_info, "user": self_label, "chatbot": qa_chat_info}

        params_ls = ["content","sessionId","message_type"]
        input_sentence = "";session_id = "0000"
        if all(param in params for param in params_ls):
            input_sentence = params["content"]
            session_id = params["sessionId"]
            message_type = params["message_type"]

            if params.has_key("user_label"):
                try:
                    if isinstance(params["user_label"],dict):
                        user_info = params["user_label"]
                    else:
                        user_info_tmp = json.loads(params["user_label"])
                        if isinstance(user_info_tmp,dict):
                            user_info = user_info_tmp
                except:
                    pass
            chatbot_info_tmp = params.get("pet_label",{})
            if isinstance(chatbot_info_tmp,dict):
                chatbot_info = chatbot_info_tmp
            else:
                try:
                    chatbot_info_tmp = json.loads(params["pet_label"])
                    if isinstance(chatbot_info_tmp,dict):
                        chatbot_info = chatbot_info_tmp
                except:
                    chatbot_info = {}
            logger.info("pet label = {}".format(chatbot_info))
            host_info_tmp = params.get("host_label",{})
            if isinstance(host_info_tmp,dict):
                host_info = host_info_tmp
            else:
                try:
                    host_info_tmp = json.loads(params["host_label"])
                    if isinstance(host_info_tmp,dict):
                        host_info = host_info_tmp
                except:
                    host_info = {}
            logger.info("host Label={}".format(host_info))
            input_sentence = remove_emoji(input_sentence)
            # logger.info("input sentence clean res{}".format(input_sentence))
            try:
                if (message_type == 1) and len(input_sentence)>0:
                    rule = NoStanfordRule(qa_host_label=host_info,qa_chat_label=chatbot_info)
                    match = Match(input_sentence,logger,session_id,rule)
                    # cache_post = requests.post(url="http://127.0.0.1", data={"userId": user_id, "petId": pet_id})
                    # if cache_post.status_code == 200:
                    #     resp = json.loads(cache_post.text)
                    #     cache = resp["cache"]

                    # if match.user_info:
                    #     for _key in match.user_info:
                    #         if type(match.user_info[_key]) in (str, int, float, unicode):
                    #             cache.set_user(_key, match.user_info[_key])
                    # for _base in base_label:
                    #     cache_user = cache.get_user(_base)
                    #     cache_host = cache.get_host(_base)
                    #     cache_chatbot = cache.get_chatbot(_base)
                    #     if cache_user:
                    #         user_info.update({_base: cache_user})
                    #     if cache_host:
                    #         host_info.update({_base: cache_host})
                    #     if cache_chatbot:
                    #         chatbot_info.update({_base: cache_chatbot})

                    # for _key in user_info:
                    #     print("the user info key= %s and value= %s" %(_key,user_info[_key]))
                    # object_info = {"host":host_info,"user":user_info,"chatbot":chatbot_info}
                    answer,answer_type,_,_,_ = main(match,proc_trie,proc_ratio,idf_trie,answer_indexs,srcFiles.template_file)

                    del match
                    del rule
                else:
                    answer = random.choice(emoji_lib)
                code = 10000
            except:
                logger.info(traceback.format_exc())
                answer = random.choice(ans_libs)
                code = 20000
        else:
            answer = "invalid params input"
            code = 30000
        t1 = time.time()
        t_delta = t1 - t0
        if isinstance(answer,unicode):
            answer = answer.encode("utf-8")
        # lock = threading.RLock()
        # lock.acquire()
        logger.info("add s,p,o into graph...")
        time_stamp = str(time.time()).replace(".","")
        add2rdf(session_id,input_sentence,answer,time_stamp)
        self.cn += 1
        if self.cn>5:
            logger.info("serailize into log file")
            pwd = cur_dir
            logger.info("current dir:\t%s" %pwd)
            if cur_dir.endswith("src"):
                #返回上级目录
                pwd = os.path.dirname(cur_dir)
                if pwd.endswith("chatbot"):
                    pwd = os.path.join(pwd,"log")
            g.serialize(os.path.join(pwd,"session.rdf"),format="xml")
            # logger.info("serailize sucessfully")
            self.cn = 0

        # lock.release()
        response = jsonify({"code":code,"response":answer,"match_type":answer_type,"time_response":t_delta})
        # response = {"code": code, "response": answer, "match_type": answer_type, "time_response": t_delta}
        return response

    # @staticmethod
    # def __init_cn():
    #     if not hasattr(g,"cn"):
    #         g.cn = 0
    #     return g.cn

def add2rdf(session_id,input_s,answer,timestamp):
    s = URIRef(urljoin(BASE_URL, str(session_id)))
    p = URIRef(urljoin(BASE_URL, "user_input_%s" %timestamp))
    o = Literal(input_s)
    g.add((s, p, o))
    p = URIRef(urljoin(BASE_URL, "response_%s" %timestamp))
    o = Literal(answer)
    g.add((s, p, o))


chatterbot = ChatterBotDouban()
chatterbot.register(app,route_base='/')

if __name__=="__main__":
    app.run('0.0.0.0',5002)
    # try:
    #     sys.exit(0)
    # except:
    #     logger.info("exit the main process and close the nlp")
    #     # nlp.close()
    #     if srcFiles.conn:
    #         srcFiles.conn.close()
    # finally:
    #     if nlp:
    #         logger.info("should close the nlp service")
    #         nlp.close()
    #     if srcFiles.conn:
    #         logger.info("should close the database connection")
    #         srcFiles.conn.close()
    #     if g:
    #         logger.info("should close graph model")
    #         g.close()

