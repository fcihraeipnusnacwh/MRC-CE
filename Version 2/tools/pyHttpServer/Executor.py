# coding=utf-8


import os, sys, urllib, time, re, subprocess
time.clock()

def Initialize():
	pass

def Run(sr):
	p = subprocess.Popen(["./SearchPhrase"] + sr.split(), stdout = subprocess.PIPE)
	out = p.communicate()
	return out[0].decode()

if __name__ == '__main__':
	print(Run('apple tree'))
	print('completed %.3f' % time.clock())