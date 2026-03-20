"""Raw UDP test - bypass pythonosc entirely"""
import socket
import time

# Test 1: Can we send/receive UDP on localhost at all?
print("=== Test 1: Basic UDP localhost ===")
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock_recv.bind(("127.0.0.1", 19999))
    sock_recv.settimeout(2.0)
    print("Receiver bound to 19999")
    
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_send.sendto(b"hello", ("127.0.0.1", 19999))
    print("Sent test packet")
    
    data, addr = sock_recv.recvfrom(1024)
    print(f"RECEIVED: {data} from {addr} - UDP WORKS!")
    sock_send.close()
except socket.timeout:
    print("TIMEOUT - UDP localhost is BROKEN (firewall?)")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    sock_recv.close()

# Test 2: Check what's listening on our ports
print("\n=== Test 2: Port check ===")
for port in [8577, 8567]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", port))
        print(f"Port {port}: FREE (nobody listening)")
        sock.close()
    except OSError as e:
        print(f"Port {port}: IN USE ({e})")

# Test 3: Send OSC-like packet to 8577 and see if anything happens
print("\n=== Test 3: Raw send to 8577 ===")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    # Minimal OSC message: /ping with no args
    # OSC format: address (null-padded to 4 bytes), type tag ","
    msg = b'/ping\x00\x00\x00,\x00\x00\x00'
    sock.sendto(msg, ("127.0.0.1", 8577))
    print("Sent raw OSC to 8577 (check server console)")
except Exception as e:
    print(f"Send failed: {e}")
finally:
    sock.close()

input("\nPress Enter to exit...")
