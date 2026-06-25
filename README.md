# guitar-karaoke

A program I made for learning to play guitar. I stopped development as a better tool already exists for this.... -> [slopsmith](https://github.com/slopsmith/slopsmith)
Unfortunately, slopsmith has been DMCA'd by Ubisoft. So I get to work on my guitar learning tool again.


Check out project writeup on [my website](https://8hu.one/projects/guitar_karaoke/index.html)

## tools used

- demucs -> htdemucs_6s and htdemucs_ft 
- ffmpeg
- whisper -> turbo model

## process

1. split the song using `./karaoke-generate`
2. play the split using `./play.py` 
2. just play the `new mkv file` generated and change audio stream to guitar karaoke or any intended stream.

## whisper.cpp

- now uses whisper.cpp for faster inference
- since my gtx 1060's cuda setup is messed up, i used the vulkan backend
