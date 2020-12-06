# !/usr/bin/python
# coding=utf-8
from __future__ import print_function,division
import jieba.posseg as psg
import json
import codecs
import jieba
import os
import re
import sys
import pdb


if sys.getdefaultencoding() != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")



# combine_rules_2gram:[(previous_flag,current_flag,after_combined_flag,slice_order,weight),...]
combine_rules_2gram = sorted([('N','N','N',1,1),('A','A','A',1,1),('D','D','D',1,1),('H','V','V',-2,1)\
,('D','A','A',-1,3),('A','N','N',1,5),('D','V','V',-1,9)\
,('uj','y','y',2,50),('uj','N','N',1,50)\
,('N','uj','uj',1,90),('A','uj','A',1,90),('*','uv','D',2,90),('ud','*','D',-2,90),('ul','*','D',-2,90)\
,('p','N','D',1,99),('V','uz','V',2,100),('V','ug','V',2,100),('V','D','V',1,100)]\
,key=lambda x:x[4])

# ,('N','A','A',-1,9)


# rectify_rules:[(current_topo,next_topo,next_next_topo,replace_position,replace_topo,weight),...]
rectify_rules = sorted([('uj','V','*',1,'N',5),('V','N','V',0,'p',99),('V','V','N',1,'p',90)],key=lambda x:x[5]) # ('A','V','*',1,'N',9),

aux_verbs = {u'能够':'can',u'能':'can',u'会':'can',
u'可':'can',u'可能':'can',u'可以':'can',u'得以':'can',u'肯':'can',
u'爱':'like',u'喜欢':'like',u'愿意':'like',u'乐意':'like',u'情愿':'like',u'请':'can',
u'讨厌':'like_not',u'厌恶':'like_not',u'拒绝':'like_not',u'反对':'like_not',
u'要':'want',u'愿':'want',u'想要':'want',u'想':'want',u'要想':'want',
u'敢':'can',u'敢于':'can',u'乐于':'like',
u'应':'should',u'应该':'should',u'应当':'should',
u'得':'should',u'该':'should',u'当':'should',u'须得':'should',
u'犯得着':'should',u'犯不着':'should_not',u'值得':'should',
u'便于':'should',u'难于':'should_not',u'难以':'should_not',u'易于':'should',
} #(type,deny_flag)

# e.g., 愿意->愿意不愿意，愿不愿意
udf_pos = {}
for k,v in aux_verbs.iteritems():
    udf_pos[k] = v
    udf_pos[u'不'+k] = (v+'_not').replace('_not_not','')
    udf_pos[u'别'+k] = (v+'_not').replace('_not_not','')
    for i in range(len(k)):
        udf_pos[k[:i+1]+u'不'+k] = v

for k,v in list(udf_pos.iteritems()):
    udf_pos[k+u'不'] = (v+'_not').replace('_not_not','')
    udf_pos[k+u'别'] = (v+'_not').replace('_not_not','')

_udf_pos = {u'给':'v',u'和':'p',u'一起':'d',u'多':'d',u'是':'B',u'身高':'n',u'上':'v'\
,u'什么时间':'d',u'什么时候':'d',u'几点':'d',u'几时':'d'\
,u'最近':'d',u'玩':'v',u'这么':'d',u'那么':'d'\
,u'爱理不理':'v',u'自由':'n',u'聊天':'v'}
udf_pos.update(_udf_pos)

qa_pos={
    u'什么':'?_what_N',
    u'怎么':'?_how_D',
    u'怎么办':'?_how_D',
    u'怎么样':'?_how_D',
    u'怎么说':'?_how_D',\
    u'如何': '?_how_D',
    u'哪儿':'?_where_N',
    u'哪':'?_where_N',
    u'哪里':'?_where_N',
    u'哪个':'?_what_N',
    u'哪些':'?_what_N',
    u'为啥':'?_why_D',
    u'为何':'?_why_D',
    u'为什么':'?_why_D',
    u'谁':'?_who_N',
    u'谁的':'?_whose_A',
    u'什么时候':'?_when_D',
    u'几时':'?_when_D',
    u'几点':'?_when_D',
    u'何时':'?_when_D',
    u'什么时间':'?_when_D'
}

udf_pos.update(qa_pos)

qa_pos_pat ={
    u'什么*':('?_what_N',['N'],'N'),
    u'哪个*':('?_what_N',['N'],'N'),
    u'哪些*':('?_what_N',['N'],'N'),
    u'啥*':('?_what_N',['N'],'N'),
    u'谁的*':('?_whose_A',['N'],'N'),
    u'几*':('?_how_many_A',['N','D'],'A'),
    u'多*':('?_how_many_A',['A','D'],'A'),
    u'怎么*':('?_how_D',['V','H','N'],'D'),
    u'咋*':('?_how_verb_D',['V'],'V')
}

N_flags = {'l':'N','q':'N','n':'N','r':'N','f':'N','s':'N'} #noun
A_flags = {'a':'A','m':'A','b':'A'} #adj
D_flags = {'d':'D','f':'D','t':'D','i':'D'} #adv
V_flags = {'v':'V'} #verb
H_flags = {'should':'H','should_not':'H','can':'H','can_not':'H','like':'H','like_not':'H'} #aux-verb
# Q_flags = {'?_what_N':'QN','?_where_N':'QN','?_who_D':'QN','?_how_many_A':'QA','?_whyD':'Q','?_how_D':'QD','?_how_verb_D':'QD'}
TOPO_flags = {}
TOPO_flags.update(N_flags)
TOPO_flags.update(A_flags)
TOPO_flags.update(D_flags)
TOPO_flags.update(V_flags)
TOPO_flags.update(H_flags)

rules_dict = {"match_rule":qa_pos_pat,"pos_rule":TOPO_flags,"combine_rule":combine_rules_2gram,
              "qa_rule":udf_pos,"change_rule":rectify_rules}

def write2json(obj,fpath):
    with codecs.open(fpath,'w') as fw:
        json.dump(obj,fw)

class MPEparse(object):
    fname = "additive_files"

    nonsense_flagCapitals = set(['x', 'w', 'y', 'z'])  # 自定义无意义分词的词性
    nonsense_flags = set(['ul', 'uz', 'ug'])
    ques_pat = re.compile(ur"[啥吗么呢？?]$|几.|[干做弄搞](嘛|什么|啥)")
    ques_pat_aux = re.compile(ur"")
    def __init__(self,match_rule=None,pos_rule=None,combine_rule=None,change_rule=None,qa_rule=None):
        self.rule_files = self.store_rules(self.fname)
        self.match_rule = self.__init_rules(self.rule_files,"match")
        if match_rule:
            self.match_rule.update(match_rule)
        self.pos_rule = self.__init_rules(self.rule_files,"pos")
        if pos_rule:
            self.pos_rule.update(pos_rule)
        self.combine_rule = self.__init_rules(self.rule_files,"combine")
        if combine_rule and (combine_rule not in self.combine_rule):
            self.combine_rule.append(combine_rule)
        self.changePos = self.__init_rules(self.rule_files,"change")
        if change_rule and (change_rule not in self.changePos):
            self.changePos.append(change_rule)
        self.qa_pos = self.__init_rules(self.rule_files,"qa")
        if qa_rule:
            self.qa_pos.update(qa_rule)
    @staticmethod
    def store_rules(fname):
        cur_dir = os.getcwd()
        if not cur_dir.endswith(fname):
            return os.path.join(os.path.dirname(cur_dir), fname)
        return cur_dir
    @classmethod
    def specify_ques(cls,s):
        if not isinstance(s,unicode):
            s = s.decode("utf")
        if s.find(u"不")>0:
            not_idx = s.find(u"不")
            if not_idx<len(s):
                not_previous = s[not_idx-1]
                not_after = s[not_idx + 1]
                if not_previous == not_after:
                    return True
        return False
    def __init_rules(self,rule_file,type):
        with codecs.open(os.path.join(rule_file,"rule.json"),'r',encoding="utf8") as fr:
            rules = json.load(fr)
            assert isinstance(rules,dict),print("type of rules is:{}".format(type(rules)))
            if type == "match":
                return rules.get("match_rule")
            elif type == "pos":
                return rules.get("pos_rule")
            elif type == "combine":
                return rules.get("combine_rule")
            elif type == "qa":
                return rules.get("qa_rule")
            elif type == "change":
                return rules.get("change_rule")
            else:
                return {}


    def __pos_shine(self,pairs_g):
        i = 0
        new_pairs = [];is_que = False;auxPos="";pair_tranPos = [];qType="";qWord=""
        pairs = list(pairs_g)
        first_aux = True
        while i<len(pairs)-1:
            qType_tmp,qWord,tmp_pair = self.__qa_pos_match(pairs[i],pairs[i+1])
            if tmp_pair:
                pairs[i] = tmp_pair
            if self.__pos2Flag(pairs[i][-1]) == "H" and first_aux:
                auxPos = pairs[i][-1]
                first_aux = False
            tmp_represent = [pairs[i][0] + ":" + pairs[i][-1],self.__pos2Flag(pairs[i][-1])]
            pair_tranPos.append(tmp_represent)
            new_pairs.append(pairs[i])
            i += 1
            if qType_tmp:
                is_que = True
                qType = qType_tmp
        if self.__pos2Flag(pairs[-1][-1]) == "H":
            auxPos = pairs[i][-1]
        pair_tranPos.append([pairs[-1][0] + ":" + pairs[-1][-1],self.__pos2Flag(pairs[-1][-1])])
        new_pairs.append(pairs[-1])
        return new_pairs,auxPos,qType,qWord,pair_tranPos,is_que
    def __qa_pos_match(self,pairCur,pairNex):
        qType,qWord,match_flag = "","",False
        for k,v in self.match_rule.iteritems():
            preffix_idx = k.find("*")
            preffix = k[:preffix_idx]
            matchType,matchFlags,transFlag = v
            if pairCur[0][:preffix_idx] == preffix:
                if pairCur[0] == preffix:
                    _next_flag = self.__pos2Flag(pairNex[-1])
                    print("next flag:",_next_flag)
                    if _next_flag in matchFlags:
                        pairCur[-1] = matchType
                        qWord = pairNex[0]
                        qType = matchType
                        match_flag = True
                elif pairCur[0] not in self.qa_pos:
                    _next_word = pairCur[0][preffix_idx:]
                    print("current word:%s and next_word %s" % (pairCur[0],_next_word))
                    _next_flag = self.__pos2Flag(jieba.user_word_tag_tab[_next_word]) if _next_word in jieba.user_word_tag_tab else "*"
                    if _next_flag in matchFlags:
                        qWord = _next_word
                        qType = matchType
                        pairCur[-1] = matchType
                        match_flag = True
                        print("qword: %s" %qWord)
                if match_flag:
                    break
        # print("pairCur flag:",pairCur[-1])
        if pairCur[-1][0] == '?' :
            qType = pairCur[-1]
            print("qType now:",qType)
        elif pairNex[-1][0] == "?":
            qType = pairNex[-1]
            print("qType now:",qType)
        return qType,qWord,pairCur

    def __pos2Flag(self,pos):
        return self.pos_rule[pos] if pos in self.pos_rule else self.pos_rule[pos[0]] if pos[0] in self.pos_rule else pos[0]

    def __getQcombine(self,pos):
        # _to_flag = self.__pos2Flag(pos)
        return pos[0] if pos.startswith("?") else pos

    def __combine_word(self,trans,is_qCombine=False):
        combine_pairs = list(trans)
        changed_cn = 0;
        while True:
            is_change = False
            for _rule in self.combine_rule:
                if not _rule:
                    continue
                assert len(_rule) == 5,print("change rule format not match:{}".format(_rule))
                _curFlag,_nexFlag,_transFlag,_order,_weight = _rule
                i = 0
                while i< len(combine_pairs) -1:
                    curFlag = combine_pairs[i][1] if not is_qCombine else self.__getQcombine(combine_pairs[i][1])
                    # print("curFlag:{} and combine_pairs[i][1]={}".format(curFlag,combine_pairs[i][1]))
                    nexFlag = combine_pairs[i+1][1] if not is_qCombine else self.__getQcombine(combine_pairs[i+1][1])
                    # print("nexFlag:{} and combine_pairs[i+1][1]={}".format(curFlag, combine_pairs[i+1][1]))
                    if (curFlag == _curFlag or (curFlag != 'x' and _curFlag == '*')) and \
                        (nexFlag == _nexFlag or (nexFlag != 'x' and _nexFlag == '*')):
                        combine_pairs[i][0] = "->".join([combine_pairs[i][0],combine_pairs[i+1][0]][::_order])
                        combine_pairs[i][1] = _transFlag
                        # pdb.set_trace()
                        combine_pairs.pop(i+1)
                        i -= 1
                        changed_cn += 1
                        is_change = True
                    i += 1
            if not is_change:
                break
        return combine_pairs,changed_cn

    def __change_flag(self,pairs):
        new_pairs = list(pairs)
        flag_changed = 0
        while True:
            is_change = False
            for _rule in self.changePos:
                i =0
                if not _rule:
                    continue
                assert len(_rule)==6,print("change rule not match result:{}".format(_rule))
                _curPos, _nexPos, _nexPos2, _replace_idx, _toPos, _ = _rule
                while i<len(new_pairs)-2:
                    curPos = new_pairs[i][-1]
                    nexPos = new_pairs[i+1][-1]
                    nexPos2 = new_pairs[i+2][-1]
                    if (curPos == _curPos or _curPos == '*') and (nexPos == _nexPos or _nexPos == '*')\
                            and (nexPos2 == _nexPos2 or _nexPos == '*'):
                        tmp_chain = new_pairs[i+_replace_idx][0]
                        new_pairs[i+_replace_idx][0] = "->".join([tmp_chain.split('->')[0].split(":")[0] + ":" + _toPos.lower()] + tmp_chain.split('->')[1:])
                        new_pairs[i+_replace_idx][1] = _toPos
                        flag_changed += 1
                        is_change = True
                    # if (nexPos.find("?")>=0 and nexPos2.startswith("V")):
                    #     new_pairs[i+2][0] = "->".join([new_pairs[i+2][0].split('->')[0].split(":")[0] + ":" + "n"] + new_pairs[i+2][0].split(
                    #         '->')[1:])
                    #     new_pairs[i + 2][1] = "N"
                    # elif nexPos2.find("?")>=0 and nexPos == 'N':
                    #     new_pairs[i+2][0] = "->".join([new_pairs[i+2][0].split('->')[0].split(":")[0] + ":" + "v"] + new_pairs[i+2][0].split(
                    #         '->')[1:])
                    #     new_pairs[i + 2][1] = 'V'
                        # print("new pairs:",new_pairs[i+2][0])
                    i += 1

                # if new_pairs[-1][-1].find("?")>=0:
                #     new_pairs[-1][0] = "->".join([new_pairs[-1][0].split('->')[0].split(":")[0] + ":" + "v"] + new_pairs[-1][0].split(
                #             '->')[1:])
                #     new_pairs[-1][0] = "V"
                # if (new_pairs[-2][-1].find("?")>=0 and new_pairs[-1][-1].lower().find("x")>=0):
                #     new_pairs[-2][0] = "->".join([new_pairs[-2][0].split('->')[0].split(":")[0] + ":" + "v"] + new_pairs[-2][0].split(
                #             '->')[1:])
                #     new_pairs[-2][1] = "V"
            if not is_change:
                break
        return new_pairs,flag_changed

    def __train_rule(self,pairs):
        new_pairs = list(pairs)
        count = 0
        while True:
            has_verb = 0
            aux_pos_idx = -1
            verb_changed_flag = 0
            new_pairs, combine_flag_1 = self.__combine_word(new_pairs, is_qCombine=0)
            new_pairs,changed_flag_1 = self.__change_flag(new_pairs)
            new_pairs, combine_flag_2 = self.__combine_word(new_pairs, is_qCombine=1)
            new_pairs,changed_flag_2 = self.__change_flag(new_pairs)
            count += 1
            print("count loop times:{}".format(count))
            for i,pair in enumerate(new_pairs):
                if pair[1] in ['V','B']:
                    has_verb = 1
                    break
                if pair[1] == 'H':
                    aux_pos_idx = i
                    has_verb = 1
                elif has_verb == 0 and aux_pos_idx >=0:
                    pair[1] = 'V'
                    verb_changed_flag = 1
                    has_verb = 1
            if has_verb == 0:
                print("length of new_pairs:%d" %len(new_pairs))
                # count_converse = 0
                for i in range(-1,-len(new_pairs)-1,-1):
                    # count_converse += 1
                    # if count_converse > 2:
                    #     break
                    if new_pairs[i][1] in ['A','p','N']:
                        print("new pairs={}".format(new_pairs[i][0]))
                        new_pairs[i][0] = "->".join(new_pairs[i][0].split('->')[:-1] + [new_pairs[i][0].split('->')[-1].split(":")[0] + ":" + 'v'])
                        break

            if changed_flag_1 + changed_flag_2 +combine_flag_1 + combine_flag_2 + verb_changed_flag == 0:
                break
        return new_pairs,has_verb

    def parse(self,sentence,verbose=True):
        if not isinstance(sentence,unicode):
            sentence = sentence.decode("utf")

        pairs = [[item.word,self.qa_pos[item.word]] if item.word in self.qa_pos else [item.word,item.flag] for item in psg.cut(sentence)]
        new_pairs, auxPos, qType, qWord, pair_tranPos,is_que = self.__pos_shine(pairs)
        print("pre ques or not:{}".format(is_que))
        print("pre qType:{}".format(qType))

        if not is_que:
            if re.search(self.ques_pat,sentence) or self.specify_ques(sentence):
                is_que = True
                qType = '?_whether_D'
                if (not qWord) and sentence.find(u"几")>0:
                    qType = '?_how_many_A'

        if verbose:
            print("ques or not:{}".format(is_que))
            print("qType:{}".format(qType))
            print("qWord info: %s" %qWord)
            print("is has aux verb : %s" %auxPos)
            print("new pairs info: ",",".join([word + "/" + flag for word,flag in pairs]))
            for word_pos,flag in pair_tranPos:
                print("word pos:%s -- flag new: %s" %(word_pos,flag))
        tranChains,has_verb = self.__train_rule(pair_tranPos)
        print("has verb: %d" %has_verb)
        for _chain,_flag in tranChains:
            print("_chain info:{} and flag:{}".format(_chain,_flag))
        qSubjects = [];qEntitys = [];qRelations = [];adjs = [];_qEntitys = []
        i = 0
        while i<len(tranChains):
            _flag = tranChains[i][-1]
            _chain = tranChains[i][0]
            if  _flag in ['V','H','B']:
                qRelations.extend([word for word in self.__get_word_by_pos(_chain,['V','A','B','H'],-1)])
                _qEntitys.extend([word for word in self.__get_word_by_pos(_chain, ['N'], -1)])
                qSubjects = _qEntitys[::-1] + qSubjects
                _qEntitys = []

            elif _flag in ['N','A','D']:
                # pdb.set_trace()
                _qEntitys.extend([word for word in self.__get_word_by_pos(_chain,['N'],-1)])
                qRelations.extend([word for word in self.__get_word_by_pos(_chain, ['V'], -1)])
                adjs.extend([word for word in self.__get_word_by_pos(_chain,['A'],-1)])

            if _chain.find("?")>0:
                if not qWord:
                    qWord = _chain.split('->')[0].split(":")[0]
                    print("_chain: {} and qWord:{}".format(_chain, qWord))
            i += 1

        if qType == '?_what_N':
            qEntitys = _qEntitys
        elif qType == '?_where_N':
            qEntitys = _qEntitys
            if has_verb == 0:
                qSubjects = _qEntitys[::-1]
                qEntitys = []
                if qWord:
                    qEntitys = [qWord]
        elif qType == '?_how_many_A':
            if has_verb == 0:
                qSubjects = _qEntitys[::-1]
                qEntitys = [qWord]
            else:
                qEntitys = _qEntitys
        elif qType in ['?_how_D','?_how_verb_D']:
            if has_verb == 0:
                qRelations += [qWord]
                qSubjects = _qEntitys[::-1]
        elif qType == '?_when_D':
            qEntitys = _qEntitys
        elif qType == '?_who_D':
            qEntitys = _qEntitys[::-1]
        elif qType == '?_whether_D':
            if not _qEntitys:
                _qEntitys = self.__chain_specify(tranChains)
            qEntitys = _qEntitys
        elif qType == '?_which_N':
            qSubjects += _qEntitys[::-1]
        else:
            print('Statements...')
            if has_verb == 0 and (not auxPos):
                qSubjects += _qEntitys[::-1]
                qWord = ([''] + adjs)[-1]
            else:
                qEntitys = _qEntitys
        qType = '' if not qType else qType[2:-2]
        qSubject = "-".join(qSubjects)
        qEntity = "-".join(qEntitys)
        qRelation = "-".join(qRelations)
        if verbose:
            print("qType:\t%s\naux_type:\t%s\nqSubject:\t%s\nqEntity:\t%s\nqRelation:\t%s\nqWord:\t%s" % \
                  (qType, auxPos, qSubject, qEntity, qRelation, qWord))
            # print("type of qSubject:{}".format(type(qSubject)))
        return (qType, auxPos, qSubject, qEntity, qRelation, qWord)

    def __chain_specify(self,tranChains):
        i = 0
        newWord = ""
        while i<len(tranChains)-1:
            curChain,curFlag = tranChains[i]
            nexChain,nexFlag = tranChains[i+1]
            if curFlag == 'B' and nexFlag == 'N':
                _chains = nexChain.split("->")
                for _chain in _chains:
                    _word = _chain.split(":")[0]
                    newWord += _word
            i += 1
        return newWord


    def __get_word_by_pos(self,chain,flags,capital_mode):
        _chains = chain.split("->")
        for _chain in _chains:
            word,pos = _chain.split(":")
            if self.__pos2Flag(pos)[capital_mode] in flags:
                yield word

    def serialize(self):
        rules = {}
        rules["match_rule"] = self.match_rule
        rules["pos_rule"] = self.pos_rule
        rules["qa_rule"] = self.qa_pos
        rules["combine_rule"] = self.combine_rule
        rules["change_rule"] = self.changePos
        with codecs.open(os.path.join(self.rule_files,"rule.json"),'w') as fw:
            json.dump(rules,fw)
    def add_rule(self,type_rule,rule_info):
        if type_rule == "combine_rule":
            self.combine_rule.append(rule_info)
        elif type_rule == "pos_rule":
            self.pos_rule.update(rule_info)
        elif type_rule == "match_rule":
            self.match_rule.update(rule_info)
        elif type_rule == "qa_rule":
            self.qa_pos.update(rule_info)
        elif type_rule == "change_rule":
            self.changePos.append(rule_info)
        else:
            pass
        # return self

def jieba_add_word():
    jieba.add_word("扩列",5,'vn')
    jieba.add_word("连麦",5,'n')
    jieba.add_word("扩约基",5,'v')
    jieba.add_word("面基",5,'v')
    jieba.add_word("约约约",5,'v')
    jieba.add_word('吃鸡',5,'vn')
    jieba.add_word("王者荣耀",5,'n')
    jieba.add_word("躺列",5,'v')
    jieba.add_word("个",5,'a')
    jieba.add_word("久",5,'a')
    jieba.add_word("次",5,'a')
    jieba.add_word("大",5,'a')
    jieba.add_word("重",5,'a')
    jieba.add_word("样",5,'n')
    jieba.add_word("家",5,'n')
    jieba.add_word("跨年",5,'vn')
    jieba.add_word("约会",10,'v')
    jieba.add_word("异地恋",5,'n')
    jieba.add_word("兴趣爱好",5,'n')
    jieba.add_word("车",10,'n')
    jieba.add_word("圣诞节",10,'n')
    jieba.suggest_freq("处男",True)
    jieba.add_word("长得",5,'v')
    jieba.suggest_freq("长得",True)
    jieba.suggest_freq(('买','房'),True)
    jieba.suggest_freq("什么时候",True)
    jieba.suggest_freq('吃鸡',True)
    jieba.suggest_freq("王者荣耀",True)
    jieba.suggest_freq("跨年",True)
    jieba.suggest_freq("躺列",True)
    jieba.suggest_freq(("我", "爱", "你"), True)
    jieba.suggest_freq("约会",True)

def mpe_add_rule(mpeParse):
    mpeParse.add_rule("combine_rule",('?','A','V',1,50))
    mpeParse.add_rule("combine_rule",('A','y','V',1,50))
    mpeParse.add_rule("match_rule",{u"是*":('?_what_N',['N','x','A'],'N')})
    mpeParse.add_rule("match_rule", {u"多少*": ('?_what_N', ['N', 'x'], 'N')})
    mpeParse.add_rule("match_rule", {u"哪*": ('?_which_N', ['N'], 'N')})
    mpeParse.add_rule("pos_rule",{"vn":"N"})
    mpeParse.add_rule("pos_rule",{"eng":"N"})
    mpeParse.add_rule("pos_rule", {"l": "V"})
    # mpeParse.add_rule("combine_rule",('N','ENG','N',1,50))
    mpeParse.add_rule("combine_rule", ('N', 'u', 'N', 1, 50))

    mpeParse.add_rule("change_rule",('N','V','A',2,'N',50))
    mpeParse.add_rule("change_rule", ('V', '?', 'A', 2, 'N', 50))
    mpeParse.add_rule("change_rule", ('N', 'u', 'A', 2, 'N', 50))
    # mpeParse.add_rule("change_rule",('*','N','?',1,'V',50))
    # mpeParse.add_rule("change_rule", ('*', 'N', 'y', 1, 'V', 50))
    # mpeParse.add_rule("change_rule", ('V', '?', '*', 1, 'N', 50))
    mpeParse.add_rule("change_rule",('H','D','N',1,'V',50))
    # mpeParse.add_rule("change_rule", ('V', 'V', '*', 1, 'N', 50))
    mpeParse.add_rule("change_rule", ('N', 'N', 'N', 2, 'V', 50))
    # mpeParse.add_rule("change_rule", ('*', 'u', 'N', 2, 'N', 50))
    # mpeParse.add_rule("change_rule", ('*', 'u', 'ENG', 2, 'N', 50))

if __name__ == '__main__':
    # write2json(rules_dict,"../additive_files/rule.json")
    import jieba.posseg as psg
    # add_rule:pos_rule = {"y":"Y"},combine_rule=('A','Y','V',1,50)
    # mpeParse = MPEparse(pos_rule = {"y":"Y"},combine_rule=('A','Y','V',1,50))
    # mpeParse.serialize()
    # mpeParse = MPEparse(combine_rule=('A','u','N',1,50))

    jieba_add_word()
    mpeParse = MPEparse()
    mpeParse.add_rule("pos_rule",{"want":"H"})
    # mpe_add_rule(mpeParse)
    # text_cases={
    #     "what":["你喜欢吃什么食物",'你主人从事什么类型的工作?','你男朋友开的是什么品牌的椅子的车','你主人从事什么工作?','你主人喜欢看什么电影','你主人的居住地址是什么?','你前男友在哪个高中上学?'],
    #     "where":['你主人喜欢去哪儿旅行?','你主人现在在哪工作?','你主人现在在哪?'],
    #     "when":['你几点下班？','你什么时候去香港','你几时给我回复？','你啥时候出生的呀？'],
    #     "how":['你打球怎么样?','你最近过得怎么样？','你爸妈身体怎么样？'],
    #     "whether":["你现在在家吗","你作业做好了没？","你哥哥长得帅吗？","你哥哥帅吗？","还吃过晚饭啦？","你昨天去医院了？","你是男的吗？"],
    #     "how_many":["你多大了?","你今年多重？","你有过几个女朋友?","今年多大了？","你哥哥谈过几个女朋友"],
    #     "":['你好啊','晚上好','我喜欢你啊','我想跟你聊天','我刚玩这个app']
    # }
    jieba.add_word("长得",10,'v')
    jieba.suggest_freq("长得",True)
    # jieba.suggest_freq(("长得","帅"),True)
    # jieba.suggest_freq(("你","好"),True)
    # for _k in text_cases:
    #     print('*****start with %s*****\n' %_k)
    #     for _s in text_cases[_k]:
    #         mpeParse.parse(_s)
    #         print('='*30)

    mpeParse.parse("你主人有多重啊")
    mpeParse.serialize()
