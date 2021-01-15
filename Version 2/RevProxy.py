# coding=utf-8

import bottle, requests, sys
from bottle import route, template, request

address = '10.131.246.51'
port = 20002

if len(sys.argv) > 1:
	adr = sys.argv[1]
	try:
		uv = adr.split(':')
		v = int(uv[1])
		address, port = uv[0], v
	except:
		pass


app = bottle.Bottle()

@app.route('<path:path>', method='GET')
def index(path):
	domain = 'http://%s:%d%s' % (address, port, request.path)
	resp = requests.get(domain, params=request.query_string)
	return resp.content

@app.route('<path:path>', method='POST')
def index(path):
	domain = 'http://%s:%d%s' % (address, port, request.path)
	resp = requests.post(domain, data=request.body.read())
	return resp.content

#bottle.run(app, host='0.0.0.0', port=port)

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
http_server = HTTPServer(WSGIContainer(app))
http_server.listen(port) 
IOLoop.instance().start()