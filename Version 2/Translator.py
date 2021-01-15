# coding=utf-8


import os, sys, urllib, time, re
from urllib.parse import urlencode 
time.clock()

def RM(patt, sr):
	mat = re.search(patt, sr, re.DOTALL | re.MULTILINE)
	return mat.group(1) if mat else ''

import urllib.request
def GetPage(url, postdata):
	proxyinfo = '218.193.131.251:11051'
	proxy_handler = urllib.request.ProxyHandler({'http': proxyinfo})
	opener = urllib.request.build_opener(proxy_handler)
	opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.115 Safari/537.36'), \
					  ('cookie','GSP=ID=52a770c11bfcedea:LM=1417608106:S=1qD2RayxZZvKmqDE; HSID=AftmTb0JCLewWt1OD; SSID=AoesVsLZJw4e3Xqi4; APISID=YaLV2UXmnJppIuPm/A82Mbm56vi-OZIDRc; SAPISID=cYQGfgyu-33MOUpd/AibR2lXZr_5t84bPf; SID=DQAAABABAABkafOvCm5yrIPaHLvZXKFaMIM9Lc-GboLDjYy7pFn-oBRqiyF9Lz3hcPlr_LVtHGdNQHn8N1QRY567IpxZMHQ6dPFI-NIXbitCoeg9X5ZbKOiBdjKdRpGTZyXeTvapjnZOPmIcwpkr9P9XIa7287-Cu8-zPOV5hDAEYEbxf89gDY7Adi4KpGigvzKIxnEnibYLyKf6TS2rJin2lw3mOLSZo-xTwkCH4No5q2l714b26GZPHrM_UJjvisR4m37E96ycIwWCi3vgcnoIz_GzFsOEQqu6E3Msojm0hAWRY6MnEmWcCSt6G3ruQEfG7HtKmkpKdIw7I6AK-ilDM_wpHIw5JcXDty4mAhbm37UbIGUpfQ; NID=67=LKmmHsJacdQkdh0VKYzVfwJkXPNCwRa6csWDAI_T2flL0DVe3uEiLxmbnHfhPW05WtPPxwymGxrkLsOOfEecufH_yGrAMBgGV_XjllI8Jrj0E96JQY_MiqZw6TVHxuSs; GOOGLE_ABUSE_EXEMPTION=ID=de1280ebf6fb7115:TM=1426951274:C=c:IP=107.161.24.251-:S=APGng0v0u3gUu9f4tba0lL1MIYCqi9_F_w; PREF=ID=52a770c11bfcedea:U=bef9bce5f436e806:LD=en:NW=1:TM=1400235664:LM=1426951405:GM=1:SG=1:S=Fdgxe_BduxSip-Xa')]
	try:
		content = opener.open(url, urlencode({'q':postdata}).encode()).read().decode()
	except: 
		content = ''
	return content

def Initialize():
	pass

def Run(sr):
	url = 'http://translate.google.com/translate_a/single?client=t&sl=en&tl=zh-CN&hl=zh-CN&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&dt=at&ie=UTF-8&oe=UTF-8'
	content = GetPage(url, sr)
	ret = RM('"(.+?)"', content)
	return ret


if __name__ == '__main__':
	Run('apple tree')
	print('completed %.3f' % time.clock())