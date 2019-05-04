# Gesture control
Gesture music player control.

## VLC Version:

* In control.py: comment out the spotify import.  
* In VLC:  
	* go to `preferences -> all -> interface -> main interfaces`  
	check Lua and Telnet  
	* go to `preferences -> all -> interface -> main interfaces -> Lua`  
	**Lua Telnet settings:**  
		* Host: localhost  
		* Port: 4212  
		* Password: 12345678  

## Spotify Version:

Requires Spotify Premium and spotipy.
* In control.py: comment out the vlc import
* Go to https://developer.spotify.com/ and register the app
* In spotify.py: Set client_id, client_secret and username to your data
