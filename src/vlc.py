import sys
import telnetlib

PORT = 4212
PASS = b"12345678"

class VLC(object):
    def __enter__(self):
        self.tn = telnetlib.Telnet("127.0.0.1", PORT)
        self.tn.read_until(b"Password: ")
        self.tn.write(PASS +b"\n")
        self.tn.read_until(b"> ")
        return self.tn
    def __exit__(self, type, value, traceback):
        self.tn.close()

def music_toggle_play():
    with VLC() as vlc:
        vlc.write(b"pause\n")

def music_vol_up():
    with VLC() as vlc:
        vlc.write(b"volup 1\n")

def music_vol_down():
    with VLC() as vlc:
        vlc.write(b"voldown 1\n")

def music_next():
    with VLC() as vlc:
        vlc.write(b"next\n")

def music_prev():
    with VLC() as vlc:
        vlc.write(b"prev\n")

