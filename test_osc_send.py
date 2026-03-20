"""Send test OSC messages to our manual server"""
import sys
sys.path.insert(0, "X:/td/StreamDiffusion/venv/Lib/site-packages")

from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("127.0.0.1", 8577)

print("Sending test prompt...")
client.send_message("/prompt", "cyberpunk city neon rain at night")
time.sleep(1)

print("Sending test seed...")
client.send_message("/seed", 42)
time.sleep(1)

print("Sending another prompt...")
client.send_message("/prompt", "underwater coral reef bioluminescent")
time.sleep(1)

print("Done! Check the Python server console for [OSC DEBUG] messages.")
