# coding=utf-8
# need bottle library

import shutil, re
import sys, os, time
import urllib, json
import bottle
from bottle import route, template, request
time.clock()


indexhtml = '''
<html lang="zh">
<head>
<title>LJQ's Http Server</title>
<meta charset="utf-8">
<link href="//cdn.bootcss.com/bootstrap/4.0.0-alpha.2/css/bootstrap.min.css" rel="stylesheet">
<script src="//cdn.bootcss.com/jquery/2.2.1/jquery.min.js"></script>
<script src="//cdn.bootcss.com/bootstrap/4.0.0-alpha.2/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
  <div class="row">
	<hr/>
  	<h1>{{name}}</h1> 
  	<h3><small>{{desc}}</small></h3>
	<hr />
  </div>
</div>
<div class="container">
	<div class="row">
		<form class="form" role="form" method="get" action=".">
			<input type="hidden" name="inpage" value="1" />
			<div class="form-group">
				<input type="text" class="form-control" name="p" placeholder="params" value="{{inp if defined('inp') else ''}}"/>
			</div>
			<input type="submit" class="btn btn-primary" value="Submit" />
			% if defined('exams') and len(exams) > 10:
			<button type="button" class="btn" data-toggle="collapse" data-target="#examples">Show Examples</button>
			% end
		</form>
	% if defined('exams'):
	<div class="row collapse in" id="examples">
		% if len(exams) <= 10:
		<ul>
			% for item, uitem in exams:
		<li><a href="./?inpage=1&p={{uitem}}">{{item}}</a></li>
			% end
		</ul>
		% end
		% if len(exams) > 10:
			% for item, uitem in exams:
		<span><a href="./?inpage=1&p={{uitem}}">{{item}}</a></span>
			% end
		% end
	<hr />
	</div>
	% end
	
	</div>
</div>

<div class="container">
	<div class="row">
	% if defined('inp'):
		Input:
		<h3>{{inp}}</h3>
	% end
	</div>
	<div class="row">
	% if defined('ret'):
		Output:
		<h3>{{!ret}}</h3>
	% end
	</div>
</div>

</body>
</html>
'''

print(sys.argv)
if len(sys.argv) < 2:
	print('USAGE: pyHttpServer.py functionfile [port]')
	sys.exit()

functfile = sys.argv[1]
shutil.copy2(functfile, 'httpmodule.py')
import httpmodule
try:
	httpmodule.Initialize()
except:
	pass

release = True
port = 0

if len(sys.argv) >= 3:
	margs = sys.argv[2:]
	for arg in margs:
		if arg == 'debug': release = False
		if re.match('^[0-9]+$', arg): port = int(arg)

if port == 0:
	try:
		port = httpmodule.port
	except:
		print('no port defined!')
		sys.exit()

try:
	name = httpmodule.name
	desc = httpmodule.desc
except:
	name, desc = "LJQ's HTTP Server", ''

try:
	examples = [(x, urllib.parse.quote(x)) for x in httpmodule.examples]
except:
	examples = []

app = bottle.app()

@app.route('/', method=['GET', 'POST'])
def index():
	p = request.params.p
	inpage = request.params.inpage
	print(p, inpage)
	if p == '': return template(indexhtml, name=name, desc=desc, exams=examples)
	ret = httpmodule.Run(p)
	if type(ret) is type({}): 
		if not inpage: 
			if 'demo' in ret: ret.pop('demo')
			ret = json.dumps(ret, ensure_ascii=False)
		else:
			if 'demo' in ret: ret = str(ret['demo'])
			else: ret = json.dumps(ret, ensure_ascii=False)
	else: ret = str(ret)
	if not inpage: return ret
	return template(indexhtml, name=name, desc=desc, exams=examples, inp=p, ret=ret.replace('\n', '<br/>'))


from bottle import static_file
@app.route('/static/<filename:path>')
def static(filename):
    return static_file(filename, root='static/')

if not release:
	bottle.run(app, host='0.0.0.0', port=port)

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
http_server = HTTPServer(WSGIContainer(app))
http_server.listen(port)
IOLoop.instance().start()

