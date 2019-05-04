# Gesture control
Gesture music player control.

# How to run

## VLC Version:

In control.py, comment the spotify import.
In VLC:
preferences -> all -> interface -> main interfaces
check Lua and Telnet
preferences -> all -> interface -> main interfaces -> Lua
Lua Telnet:
	* Host: localhost
	* Port: 4212
	* Password: 12345678

## Spotify Version:

Requires Spotify Premium and spotipy.
Go to https://developer.spotify.com/ and register app.
Set client_id, client_secret and username to your data.