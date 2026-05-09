import json
import argparse
import sys
import wave
import numpy as np
from pathlib import Path

def ms_to_ass(ms: int) -> str:
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    cs = ms // 10
    return f"{h}:{m:02}:{s:02}.{cs:02}"

def get_ass_header() -> str:
    return """[Script Info]
Title: Lyrics
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Mochiy Pop One,72,&H001F004C,&H00FFFFFF,&H00111111,&H00111111,0,0,0,0,100,100,0,0,1,4,3,5,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

def analyze_audio(wav_path: Path, fps=25):
    with wave.open(str(wav_path), 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        if wav.getnchannels() == 2:
            audio_data = audio_data[::2] / 2 + audio_data[1::2] / 2
        chunk_size = int(wav.getframerate() / fps)
        remainder = len(audio_data) % chunk_size
        if remainder: audio_data = audio_data[:-remainder]
        chunks = audio_data.reshape(-1, chunk_size).astype(np.float64)
        rms = np.sqrt(np.mean(chunks**2, axis=1))
        max_rms = np.max(rms)
        return (rms / max_rms if max_rms > 0 else rms), fps

def smooth_data(data, alpha=0.3):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def get_vocal_color(v_rms: float):
    r = int(76 + (179 * v_rms))
    b = int(31 + (71 * v_rms))
    return f"&H00{b:02X}00{r:02X}&"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()

    json_input = next(args.target_dir.glob("*.json"), None)
    stem_dir = args.target_dir / "demucs_ft"
    v_wav, b_wav, d_wav = stem_dir/"vocals.wav", stem_dir/"bass.wav", stem_dir/"drums.wav"
    output_ass = args.target_dir / "vocals.ass"

    if not all([json_input, v_wav.exists(), b_wav.exists(), d_wav.exists()]):
        print("Error: Missing JSON or Stems (bass/drums/vocals). Check directory.")
        sys.exit(1)

    print(f"Crunching Stems for {args.target_dir.name}...")
    v_rms, fps = analyze_audio(v_wav)
    b_rms, _ = analyze_audio(b_wav)
    d_rms, _ = analyze_audio(d_wav)
    
    v_rms_smoothed = smooth_data(v_rms, alpha=0.25)
    ms_per_frame = int(1000 / fps)

    with json_input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ass_lines = [get_ass_header()]
    valid_lines = []
    
    for seg in data.get("transcription", []):
        tokens = [tok for tok in seg.get("tokens", []) if not tok["text"].startswith("[_")]
        if tokens: 
            valid_lines.append({
                "start": seg["offsets"]["from"], 
                "end": seg["offsets"]["to"], 
                "tokens": tokens
            })

    for idx, line in enumerate(valid_lines):
        line_start = line["start"]
        prev_start = valid_lines[idx-1]["start"] if idx > 0 else max(0, line_start - 2000)
        next_start = valid_lines[idx+1]["start"] if idx < len(valid_lines) - 1 else line["end"] + 3000

        event_start, event_end = prev_start, next_start + 400
        rel_drop_start = line_start - event_start
        rel_drop_end = rel_drop_start + 300 

        line_text = (f"{{\\move(960,820,960,950,{rel_drop_start},{rel_drop_end})"
                     f"\\fad(400,400)\\alpha&H66&\\t({rel_drop_start},{rel_drop_end},\\alpha&H00&)}}")

        line_frames_start = int(event_start / ms_per_frame)
        line_frames_end = int(event_end / ms_per_frame)
        
        for i, f_idx in enumerate(range(line_frames_start, line_frames_end)):
            idx = max(0, min(f_idx, len(b_rms)-1))
            direction = 1 if i % 2 == 0 else -1
            shear_x = round((b_rms[idx] * 0.06) * direction, 3)
            stretch_y = int(100 + (d_rms[idx] * 35))
            blur = round(b_rms[idx] * 1.5, 1)

            t_start = (f_idx * ms_per_frame) - event_start
            t_end = t_start + ms_per_frame
            line_text += f"{{\\t({t_start},{t_end},\\fax{shear_x}\\fscy{stretch_y}\\blur{blur})}}"

        for tok in line["tokens"]:
            t0, t1 = tok["offsets"]["from"], tok["offsets"]["to"]
            word_frames_start, word_frames_end = int(t0 / ms_per_frame), int(t1 / ms_per_frame)
            
            word_tags = ""
            for f_idx in range(word_frames_start, word_frames_end):
                idx = max(0, min(f_idx, len(v_rms_smoothed)-1))
                c1 = get_vocal_color(v_rms_smoothed[idx])
                
                t_start = (f_idx * ms_per_frame) - event_start
                t_end = t_start + ms_per_frame
                word_tags += f"\\t({t_start},{t_end},\\1c{c1})"
            
            settle = (word_frames_end * ms_per_frame) - event_start
            word_tags += f"\\t({settle},{settle+200},\\1c&H001F004C&)"
            
            line_text += f"{{{word_tags}}}{tok['text']}"

        ass_lines.append(f"Dialogue: 0,{ms_to_ass(event_start)},{ms_to_ass(event_end)},Default,,0,0,0,,{line_text}")

    with output_ass.open("w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))
    print(f"Success! {output_ass.name} generated in {args.target_dir}.")

if __name__ == "__main__":
    main()
