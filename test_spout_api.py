"""Check SpoutGL API"""
import SpoutGL
print("SpoutGL version:", getattr(SpoutGL, '__version__', 'unknown'))
print()
print("All attributes:")
for attr in sorted(dir(SpoutGL)):
    if not attr.startswith('_'):
        print(f"  {attr}")
print()
print("SpoutSender methods:")
s = SpoutGL.SpoutSender()
for attr in sorted(dir(s)):
    if not attr.startswith('_'):
        print(f"  {attr}")
