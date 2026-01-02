
import os
import json
import time
import webbrowser
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

# Constants
TOKEN_FILE = r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json"
LOCAL_REDIRECT_HOST = "127.0.0.1"
LOCAL_REDIRECT_PORT = 8000
LOCAL_REDIRECT_URL = f"http://{LOCAL_REDIRECT_HOST}:{LOCAL_REDIRECT_PORT}"

# Token capture handler
class _TokenCaptureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        token = qs.get("request_token", [None])[0]
        if token:
            self.server.request_token = token
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>Request token received.</h3><p>You can close this window.</p>")
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>No request_token found.</h3><p>Please copy it manually.</p>")

def capture_request_token_via_local_server(timeout=60):
    server_address = (LOCAL_REDIRECT_HOST, LOCAL_REDIRECT_PORT)
    httpd = HTTPServer(server_address, _TokenCaptureHandler)
    httpd.request_token = None

    def _serve():
        end_time = time.time() + timeout
        while time.time() < end_time and httpd.request_token is None:
            httpd.handle_request()

    t = Thread(target=_serve, daemon=True)
    t.start()
    t.join(timeout)
    return httpd.request_token

# Ensure token file exists
if not os.path.exists(TOKEN_FILE):
    print("⚠️ Token file not found. Creating a new one...")
    api_key = input("Enter your Kite API key: ").strip()
    with open(TOKEN_FILE, "w") as f:
        json.dump({"api_key": api_key}, f, indent=2)

# Load token data
with open(TOKEN_FILE, "r") as f:
    token_data = json.load(f)

api_key = token_data.get("api_key")
if not api_key:
    raise SystemExit("API key missing in token file. Please add it.")

kite = KiteConnect(api_key=api_key)

# Always force refresh
print("🔄 Forcing token refresh...")
login_url = kite.login_url()
print(f"Opening Kite login URL: {login_url}")
try:
    webbrowser.open(login_url, new=2)
except Exception:
    print("Please open this URL manually:", login_url)

print(f"Attempting to capture request_token at {LOCAL_REDIRECT_URL} (timeout 60s).")
request_token = capture_request_token_via_local_server(timeout=60)
if not request_token:
    print("Automatic capture failed. Paste request_token from browser redirect URL.")
    request_token = input("Paste request_token here: ").strip()

api_secret = input("Enter your Kite API secret: ").strip()

try:
    new_data = kite.generate_session(request_token, api_secret=api_secret)
    new_access_token = new_data.get("access_token")
    if not new_access_token:
        raise RuntimeError("No access_token returned from generate_session().")

    kite.set_access_token(new_access_token)
    profile = kite.profile()
    user_name = profile.get("user_name", "unknown")

    token_data.update({"access_token": new_access_token, "user_name": user_name})
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"✅ New access token saved to {TOKEN_FILE}")
    print(f"✅ Logged in as: {user_name}")

except Exception as e:
    raise SystemExit("Failed to generate session / renew token: " + str(e))
