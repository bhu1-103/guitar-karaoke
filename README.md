# guitar-karaoke

A program I made for learning to play guitar. I stopped development as a better tool already exists for this.... -> [slopsmith](https://github.com/slopsmith/slopsmith)
Unfortunately, slopsmith has been DMCA'd by Ubisoft. So I get to work on my guitar learning tool again.

July 06 -> I started working on a DIY AMP and Pedal

Check out project writeup on [my website](https://8hu.one/projects/guitar_karaoke/index.html)

## tools used

### software 

#### for backing track

- demucs -> htdemucs_6s and htdemucs_ft 
- ffmpeg
- whisper -> turbo model

#### for the hardware

- guitarix
- jack

### hardware
- Audio Interface -> Behringer UM2
- Raspberry Pi 5 with 10k mah power bank and a 3.5 inch display to replace an amp knobs
- My Super-Strat

## process

1. split the song using `./karaoke-generate`
2. play the split using `./play.py` 
2. just play the `new mkv file` generated and change audio stream to guitar karaoke or any intended stream.

## whisper.cpp

- now uses whisper.cpp for faster inference
- since my gtx 1060's cuda setup is messed up, i used the vulkan backend
