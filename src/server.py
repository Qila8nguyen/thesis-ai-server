from http.server import CGIHTTPRequestHandler, HTTPServer
import cgi, json

class MyServer(CGIHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)    
    self.end_headers()
    self.wfile.write(b'Hello, World!')
  def do_POST(self):
    # Parse the form data
    form = cgi.FieldStorage(
        fp=self.rfile,
        headers=self.headers,
        environ={'REQUEST_METHOD': 'POST',
                  'CONTENT_TYPE': self.headers['Content-Type'],
                  }
    )
    # Get the form values
    domain = form.getvalue("domain")
    professional = form.getvalue("professional")
    path = self.path
    self.send_response(200)
    self.end_headers()

    print(form)
    print(domain)
    print(professional)
    response = 'POST! '+ path
    self.wfile.write(response.encode())