# guitar-karaoke

A program I made for learning to play guitar. I stopped development as a better tool already exists for this.... -> [slopsmith](https://github.com/slopsmith/slopsmith)

## tools used

- demucs -> htdemucs_6s and htdemucs_ft 
- ffmpeg
- whisper -> small model

## process

- split the song using `./karaoke-generate`
- play the split using `./play.py`

## whisper.cpp

- now uses whisper.cpp for faster inference
- since my gtx 1060's cuda setup is messed up, i used the vulkan backend
