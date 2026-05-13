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

def get_minimal_ass_header() -> str:
    return """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Mochiy Pop One,68,&H00FFFFFF,&H00FFFFFF,&H00111111,&H00111111,0,0,0,0,100,100,0,0,1,4,3,5,40,40,60,1
"""

def analyze_audio(wav_path: Path, fps=60):
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

def get_pulse_color(v_rms: float):
    val = int(130 + (v_rms * 125))
    return f"&H{val:02X}{val:02X}{val:02X}&"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()

    json_input = next(args.target_dir.glob("*.json"), None)
    stem_dir_1 = args.target_dir / "demucs_ft"
    stem_dir_2 = args.target_dir / "demucs_6s"
    v_wav = stem_dir_1 / "vocals.wav"
    b_wav = stem_dir_1 / "bass.wav"
    d_wav = stem_dir_1 / "drums.wav"
    g_wav = stem_dir_2 / "guitar.wav"
    output_ass = args.target_dir / "vocals.ass"

    if not all([json_input, v_wav.exists(), b_wav.exists(), d_wav.exists(), g_wav.exists()]):
        print("Error: Missing JSON or Stems. Check directory.")
        sys.exit(1)

    v_rms, fps = analyze_audio(v_wav)
    b_rms, _ = analyze_audio(b_wav)
    d_rms, _ = analyze_audio(d_wav)
    g_rms, _ = analyze_audio(g_wav)
    
    v_rms_smoothed = smooth_data(v_rms, alpha=0.5)
    ms_per_frame = int(1000 / fps)

    with json_input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    current_line_words = []
    global_nudge = -300 # negative delay for karaoke
    
    for seg in data.get("transcription", []):
        clean_text = seg.get("text", "").replace("\n", "").strip()
        if not clean_text:
            continue
            
        start_time = seg["offsets"]["from"] + global_nudge
        end_time = seg["offsets"]["to"] + global_nudge
        
        word_obj = {
            "text": clean_text,
            "start": start_time,
            "end": end_time
        }
        
        if current_line_words:
            prev_end = current_line_words[-1]["end"]
            gap = start_time - prev_end
            ends_with_punct = current_line_words[-1]["text"].endswith(('.', '!', '?', ','))
            
            if gap > 800 or len(current_line_words) >= 8 or ends_with_punct:
                lines.append(current_line_words)
                current_line_words = []
                
        current_line_words.append(word_obj)

    if current_line_words:
        lines.append(current_line_words)

    ass_lines = [get_minimal_ass_header(), "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    active_color = "&H6600FF&"
    for line in lines:
        if not line: continue
        
        line_start = line[0]["start"]
        line_end = line[-1]["end"]
        
        event_start = max(0, line_start - 200)
        event_end = line_end + 400
        
        rel_drop_start = line_start - event_start
        rel_drop_end = rel_drop_start + 200 

        line_frames_start = int(event_start / ms_per_frame)
        line_frames_end = int(event_end / ms_per_frame)
        
        l0_tags, l1_tags, l2_tags, l3_tags, pulse_tags = "", "", "", "", ""
        
        for f_idx in range(line_frames_start, line_frames_end):
            idx_rms = max(0, min(f_idx, len(b_rms)-1))
            v_idx = max(0, min(f_idx, len(v_rms_smoothed)-1))
            
            bass_bump = round(b_rms[idx_rms] * 15, 1)
            stretch_y = int(100 + (d_rms[idx_rms] * 35))
            guitar_val = g_rms[idx_rms]
            aberration = round(guitar_val * 1.0, 6) 
            
            t_start = (f_idx * ms_per_frame) - event_start
            
            l0_tags += f"\\t({t_start},{t_start},\\fax{-aberration}\\fscy{stretch_y}\\bord{3 + bass_bump})"
            l1_tags += f"\\t({t_start},{t_start},\\fax{aberration}\\fscy{stretch_y}\\bord{3 + bass_bump})"
            l2_tags += f"\\t({t_start},{t_start},\\fscy{stretch_y}\\bord{1.5 + bass_bump})"
            l3_tags += f"\\t({t_start},{t_start},\\fscy{stretch_y}\\bord{2.5 + bass_bump})"
            
            c_pulse = get_pulse_color(v_rms_smoothed[v_idx])
            pulse_tags += f"\\t({t_start},{t_start},\\1c{c_pulse})"

        common_prefix = f"{{\\q2\\move(960,820,960,950,{rel_drop_start},{rel_drop_end})\\fad(200,200)"
        
        l0_text = f"{common_prefix}\\alpha&H44&\\1c&H0000FF&\\3c&H0000FF&\\blur3{l0_tags}}}"
        l1_text = f"{common_prefix}\\alpha&H44&\\1c&HFFFF00&\\3c&HFFFF00&\\blur3{l1_tags}}}"
        l2_text = f"{common_prefix}\\3c&H000000&\\blur0{l2_tags}{pulse_tags}}}"
        l3_text = f"{common_prefix}\\alpha&HFF&\\1c{active_color}\\3c&H000000&\\blur0{l3_tags}}}"
        
        for i, word in enumerate(line):
            t0 = word["start"] - event_start
            t1 = word["end"] - event_start
            
            pop_anim = f"\\t({t0},{t0},\\fscx130)\\t({t1},{t1},\\fscx100)"
            word_txt = word['text']
            
            # Add a space before the ASS tags if it's not the first word
            space = " " if i > 0 else ""
            
            l0_text += f"{space}{{{pop_anim}}}{word_txt}"
            l1_text += f"{space}{{{pop_anim}}}{word_txt}"
            l2_text += f"{space}{{{pop_anim}}}{word_txt}"
            l3_text += f"{space}{{\\alpha&HFF&\\t({t0},{t0},\\alpha&H00&\\fscx130)\\t({t1},{t1},\\alpha&HFF&\\fscx100)}}{word_txt}"

        ass_lines.append(f"Dialogue: 0,{ms_to_ass(event_start)},{ms_to_ass(event_end)},Default,,0,0,0,,{l0_text}")
        ass_lines.append(f"Dialogue: 1,{ms_to_ass(event_start)},{ms_to_ass(event_end)},Default,,0,0,0,,{l1_text}")
        ass_lines.append(f"Dialogue: 2,{ms_to_ass(event_start)},{ms_to_ass(event_end)},Default,,0,0,0,,{l2_text}")
        ass_lines.append(f"Dialogue: 3,{ms_to_ass(event_start)},{ms_to_ass(event_end)},Default,,0,0,0,,{l3_text}")

    with output_ass.open("w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))

if __name__ == "__main__":
    main()
