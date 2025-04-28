#!/usr/bin/env python3

import os
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer

RESIZED_DIR = 'demo_videos'  # The folder containing your 224×224 .mp4 files
PORT = 10001

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/videos':
            # Return a JSON list of all .mp4 files under RESIZED_DIR
            video_files = []
            base_path = os.path.join(os.getcwd(), RESIZED_DIR)

            for root, dirs, files in os.walk(base_path):
                for f in files:
                    if f.lower().endswith('.mp4'):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, os.getcwd())
                        # e.g. "demo_videos_resized/video01.mp4"
                        # We want to serve it from "/demo_videos_resized/video01.mp4"
                        web_path = '/' + rel_path.replace('\\', '/')
                        video_files.append(web_path)

            # Return JSON array
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(video_files).encode('utf-8'))
        
        else:
            # Serve static files (index.html, the .mp4, etc.)
            super().do_GET()

def run_server(port=PORT):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"Serving on http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
