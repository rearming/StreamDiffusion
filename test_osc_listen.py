"""Quick OSC listener test - run this and change params in TD to see if anything arrives"""
import sys, time
sys.path.insert(0, "X:/td/StreamDiffusion/venv/Lib/site-packages")

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading

def catch_all(address, *args):
    print(f"[RECEIVED] {address} = {args}")

dispatcher = Dispatcher()
dispatcher.set_default_handler(catch_all)

# Try BOTH ports
for port in [8577, 8567, 8247, 8248]:
    try:
        server = BlockingOSCUDPServer(("127.0.0.1", port), dispatcher)
        print(f"Listening on port {port}...")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    except Exception as e:
        print(f"Port {port}: {e}")

print("\nWaiting for OSC messages from TD... Change a parameter in TouchDesigner.")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")
