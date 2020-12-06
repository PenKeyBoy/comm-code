# !/usr/bin/python
# coding=utf-8
from __future__ import print_function
from urllib import urlopen
import requests
import traceback
import time
from bs4 import BeautifulSoup
import random
import json
import sys
import execjs

if sys.getdefaultencoding()!="utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")

proxy_pools = []
headers = {
    "Host":"translate.google.cn",
    "connection":"keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate,br",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7"
}
main_url = "https://translate.google.cn/?hl=en&op=translate&sl=auto&tl=en&text=%s"

# url = "https://translate.google.cn/translate_a/single?client=webapp&sl=auto&tl=en&hl=en&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&dt=gt&ssel=0&tsel=0&kc=1&tk=463172.99603&q=%s"
def get_proxy(proxy_url):
    proxy_ls = []
    while True:
        try:
            proxy_ls = list(set(urlopen(proxy_url).read().split("\n")[:-1]))
            break
        except Exception:
            traceback.format_exc()
            time.sleep(5)
            continue
    return proxy_ls

def get_match_english(url,tk,text):
    google_url = url %(tk,text)
    proxy = {}
    proxy["http"] = "http://" + random.choice(get_proxy('http://60.205.92.109/api.do?name=B1B96B092E52FED2DF45618614C3A20A&status=1&type=static'))
    # print("current proxy result:\n",proxy)
    try:
        req = requests.get(google_url,headers=headers,proxies=proxy,timeout=1,verify=False)
        if req.status_code == 200:
            soup = BeautifulSoup(req.content,"lxml")

            # answer_rule = "div > span.tlid-translation.translation > span"

            match_res = select_matchest(soup,"p")
            if match_res:
                # print("translate res:{} and res for radio:{}".format(match_res[0],match_res[-1]))
                return match_res
        else:
            print("response fail status_code=%d and response=%s" %(req.status_code,req.content))
            return ""
    except:
        print(traceback.format_exc())
        return

def select_matchest(soup,rule):
    radio_chineses=[]
    answer_tags = soup.select(rule)
    if answer_tags:
        # print("answer tags length=%d" %len(answer_tags))
        answer_text = answer_tags[0].text
        # trs = soup.select("tr.gt-baf-entry.gt-baf-entry-selected")
        ls = json.loads(answer_text)
        assert isinstance(ls,list)
        # print("ls len=%d" %len(ls))
        if len(ls)>=2:
            translate_info = ls[0][0][0]
            # print("translate info:",translate_info)
            try:
                word_info = ls[1][0][2]
            except:
                return translate_info,""
            for _radio in word_info:
                # assert len(_radio)==2,print("not match info:",_radio)
                if _radio[0].lower() == translate_info.lower():
                    if isinstance(_radio[1],list):
                        # print(_radio[1])
                        res = "|".join(_radio[1])
                        if isinstance(res,unicode):
                            res = res.encode("utf8")
                        return translate_info,res
                    else:
                        print("not list res:",_radio[1])
                        return translate_info,""
    return "",""

#compute tk value

class TK():
    def __init__(self):
        self.exc = execjs.compile("""
        function TL(a) {
        var k = "";
        var b = 406644;
        var b1 = 3293161072;

        var jd = ".";
        var $b = "+-a^+6";
        var Zb = "+-3^+b+-f";

        for (var e = [], f = 0, g = 0; g < a.length; g++) {
            var m = a.charCodeAt(g);
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
            e[f++] = m >> 18 | 240,
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
            e[f++] = m >> 6 & 63 | 128),
            e[f++] = m & 63 | 128)
        }
        a = b;
        for (f = 0; f < e.length; f++) a += e[f],
        a = RL(a, $b);
        a = RL(a, Zb);
        a ^= b1 || 0;
        0 > a && (a = (a & 2147483647) + 2147483648);
        a %= 1E6;
        return a.toString() + jd + (a ^ b)
    };

    function RL(a, b) {
        var t = "a";
        var Yb = "+";
        for (var c = 0; c < b.length - 2; c += 3) {
            var d = b.charAt(c + 2),
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
        }
        return a
    }
    """)

    def getTK(self,text):
        return self.exc.call("TL",text)

if __name__ == '__main__':

    tk = TK()
    tk_val = tk.getTK("电影作品")
    print("tk_val:",tk_val)
    url = "https://translate.google.cn/translate_a/single?client=webapp&sl=auto&tl=en&hl=en&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&dt=gt&ssel=0&tsel=0&kc=1&tk=%s&q=%s"
    trans, chinese_radio = get_match_english(url,tk_val, "电影作品")
    print("tans res:%s and radio res: %s" %(trans,chinese_radio))
    # assert str(tk_val)=="437271.14400",print("long tk value:",tk_val)




