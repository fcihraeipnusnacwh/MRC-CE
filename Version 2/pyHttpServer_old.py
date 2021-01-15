# coding=utf-8

import socketserver
import shutil
import sys, os, time
import urllib
from http import server
time.clock()


indexhtml = '''
<html>
<head><title>LJQ's Http Server</title></head>
<body>
<form method="post" action=".">
<input type="text" name="p"/>
<br/>
<input type="submit" name="submit" />
</form>
</body>
</html>
'''


if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print('USAGE: pyHttpServer.py functionfile port')
	else:
		port = int(sys.argv[2])
		functfile = sys.argv[1]
		shutil.copy2(functfile, 'httpmodule.py')
		import httpmodule
		httpmodule.Initialize()

		class MyHttpHandler(server.BaseHTTPRequestHandler):
			def Process(self, queryString):
				params = urllib.parse.parse_qs(queryString)     
				#print( params )
				return httpmodule.Run( params['p'][0] )

			def do_GET(self):
				print(self.path)
				ppr = self.path.split('?', 1)
				self.send_response(200)
				self.send_header("Content-type", "text/html")
				self.end_headers()
				if len(ppr) == 1: 
					self.wfile.write(indexhtml.encode())
					return 
				queryString = urllib.parse.unquote(ppr[1])
				html = self.Process(queryString)
				self.wfile.write(html.encode())

			def do_POST(self):
				self.send_response(200)
				self.send_header("Content-type", "text/html")
				self.end_headers()
				length = int(self.headers['Content-Length'])
				queryString = self.rfile.read(length).decode('utf-8')
				#print(queryString)
				html = self.Process(queryString)
				self.wfile.write(html.encode())

		httpd = server.HTTPServer(('',port), MyHttpHandler)
		print('Starting simple httpd on port: ' + str(httpd.server_port))
		httpd.serve_forever()
	print('completed %.3f' % time.clock())