import spotipy
import spotipy.util as util

client_id = '57583cfec79b460086a6d1797e37b4e6'
client_secret = 'af01a3d3668546c69af4d80081f3b0e6'
username = 'qrog3t1yh7hjxlgmrl1v935sp'


redirect_url = 'http://localhost:8888/callback/'
scope = '''user-read-playback-state user-modify-playback-state
 user-read-currently-playing playlist-read-private user-read-recently-played'''

def music_toggle_play():
    devices = sp.devices()['devices']
    if(sp.current_playback()['is_playing']):
        for dev in devices:
            dev_id = dev['id']
            sp.pause_playback(dev_id)
    else:
        for dev in devices:
            dev_id = dev['id']
            sp.start_playback(dev_id)

def music_vol_up():
    devices = sp.devices()['devices']
    for dev in devices:
        dev_id = dev['id']
        vol = dev['volume_percent']
        vol += 10
        if vol > 100:
            vol = 100
        sp.volume(vol, dev_id)

def music_vol_down():
    devices = sp.devices()['devices']
    for dev in devices:
        dev_id = dev['id']
        vol = dev['volume_percent']
        vol += 10
        if vol < 0:
            vol = 0
        sp.volume(vol, dev_id)

def music_next():
    devices = sp.devices()['devices']
    for dev in devices:
        dev_id = dev['id']
        sp.next_track(dev_id)
    pass

def music_prev():
    devices = sp.devices()['devices']
    for dev in devices:
        dev_id = dev['id']
        sp.previous_track(dev_id)
    pass

token = util.prompt_for_user_token(
        username=username,
        scope=scope,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_url)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print('ERROR: can\'t get token')