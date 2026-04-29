import pygame
import numpy as np
import librosa
import math
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python play.py <song name>")
    sys.exit(1)

SONG = sys.argv[1]
BASE = os.path.expanduser(f"~/Music/karaoke/cache/{SONG}")
FT = f"{BASE}/demucs_ft"
S6 = f"{BASE}/demucs_6s"
SRT = f"{BASE}/vocals.srt"

W, H = 1600, 900
FPS = 60
SR = 44100
HOP = SR // FPS

# ── load stems ────────────────────────────────────────────────────────
print("Loading stems...")
stem_names = ["other", "bass", "drums", "guitar", "vocals", "piano"]
stem_paths = {
    "other":  f"{S6}/other.wav",
    "bass":   f"{FT}/bass.wav",
    "drums":  f"{FT}/drums.wav",
    "guitar": f"{S6}/guitar.wav",
    "vocals": f"{FT}/vocals.wav",
    "piano":  f"{S6}/piano.wav",
}
stems = {}
for name, path in stem_paths.items():
    y, _ = librosa.load(path, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    stems[name] = y
    print(f"  loaded {name}")

n_samples = min(s.shape[1] for s in stems.values())

# ── mix presets ───────────────────────────────────────────────────────
MIXES = {
    "z": {"name": "Guitar Karaoke",    "on": ["other","bass","drums","piano","vocals"]},
    "x": {"name": "Backing Track",     "on": ["other","bass","drums","piano"]},
    "c": {"name": "Vocals Karaoke",    "on": ["other","bass","drums","piano","guitar"]},
    "v": {"name": "Original",          "on": ["other","bass","drums","guitar","vocals","piano"]},
}

stem_enabled = {n: True for n in stem_names}

def build_mix(enabled):
    mixed = np.zeros((2, n_samples), dtype=np.float64)
    for name, on in enabled.items():
        if on:
            mixed += stems[name][:, :n_samples]
    mx = np.abs(mixed).max() + 1e-6
    return (mixed / mx * 0.8).T.astype(np.float32)

def make_sound(enabled):
    arr = build_mix(enabled)
    arr_int16 = (arr * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(arr_int16))

# ── parse srt ─────────────────────────────────────────────────────────
def parse_srt(path):
    subs = []
    if not os.path.exists(path):
        return subs
    with open(path) as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            times = lines[1].split(" --> ")
            def to_sec(t):
                t = t.strip().replace(",", ".")
                h, m, s = t.split(":")
                return int(h)*3600 + int(m)*60 + float(s)
            start = to_sec(times[0])
            end = to_sec(times[1])
            text = " ".join(lines[2:])
            subs.append((start, end, text))
        except:
            continue
    return subs

subs = parse_srt(SRT)

def get_subtitle(t):
    for start, end, text in subs:
        if start <= t <= end:
            return text
    return ""

def render_text_wrapped(text, font, color, max_width):
    lines = []
    paragraphs = text.split('\n')
    
    for p in paragraphs:
        current_line = ""
        for char in p:
            test_line = current_line + char
            fw, fh = font.size(test_line)
            if fw <= max_width:
                current_line = test_line
            else:
                last_space = current_line.rfind(' ')
                if last_space != -1 and not char.isspace():
                    lines.append(current_line[:last_space])
                    current_line = current_line[last_space+1:] + char
                else:
                    lines.append(current_line)
                    current_line = char
        if current_line:
            lines.append(current_line)

    surfaces = [font.render(line, True, color) for line in lines]
    return surfaces, font.get_linesize()

# ── guitar strings beyond frame ───────────────────────────────────────
def get_strings():
    strings = []
    for i in range(6):
        t = i / 5
        x1 = int(W * (0.7 + t * 0.4))   
        y1 = int(H * (-0.1 + t * 0.15)) 
        x2 = int(W * (-0.1 + t * 0.15)) 
        y2 = int(H * (0.8 + t * 0.3))    
        strings.append((x1, y1, x2, y2))
    return strings

STRINGS = get_strings()

# ── particles & effect states ─────────────────────────────────────────
N_PARTICLES = 120
px = np.random.uniform(0, W, N_PARTICLES)
py = np.random.uniform(0, H, N_PARTICLES)
pvx = np.random.uniform(-0.5, 0.5, N_PARTICLES)
pvy = np.random.uniform(-0.5, 0.5, N_PARTICLES)

drum_impacts = []  
prev_drum_amp = 0.0
bass_shockwaves = [] 
prev_bass_amp = 0.0

# ── pygame init ───────────────────────────────────────────────────────
pygame.mixer.init(frequency=SR, size=-16, channels=2)
pygame.init()

screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.display.set_caption(f"♪ {SONG}")
clock = pygame.time.Clock()

# ── Font Loading ──────────────────────────────────────────────────────
font = pygame.font.SysFont("monospace", 18)
font_large = pygame.font.SysFont("monospace", 28, bold=True)

# 1. Load your custom English Font
brico_path = os.path.expanduser("~/.local/share/fonts/BricolageGrotesque-Regular.ttf")
try:
    font_eng = pygame.font.Font(brico_path, 46)
except:
    print("Bricolage Grotesque not found, falling back.")
    font_eng = pygame.font.SysFont("sans", 46, bold=True)

font_jap = pygame.font.SysFont("notoserifcjkjp", 46, bold=True)

def has_japanese(text):
    """Checks if text contains Hiragana, Katakana, or Kanji"""
    for char in text:
        if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF':
            return True
    return False

# ── initial sound ─────────────────────────────────────────────────────
current_sound = make_sound(stem_enabled)
current_sound.play()
start_ticks = pygame.time.get_ticks()
paused = False
pause_offset = 0
seek_offset = 0  
show_lyrics = True
mode = "mix"  
status_msg = ""
status_timer = 0

def get_pos_samples():
    if paused:
        return pause_offset + seek_offset
    elapsed_ms = pygame.time.get_ticks() - start_ticks
    return int((elapsed_ms / 1000.0) * SR) + seek_offset

def restart_playback(pos_samples):
    global current_sound, start_ticks, seek_offset
    current_sound.stop()
    seek_offset = pos_samples
    trimmed = build_mix(stem_enabled)
    trimmed = trimmed[pos_samples:]
    arr_int16 = np.ascontiguousarray((trimmed * 32767).astype(np.int16))
    current_sound = pygame.sndarray.make_sound(arr_int16)
    start_ticks = pygame.time.get_ticks()
    if not paused:
        current_sound.play()

def get_chunk_mono(name, pos):
    s = stems[name][0]
    c = s[pos:pos + HOP]
    if len(c) < HOP:
        c = np.pad(c, (0, HOP - len(c)))
    return c

def amplitude(c):
    return float(np.sqrt(np.mean(c**2)))

def fft_bins(c, n=64):
    f = np.abs(np.fft.rfft(c, n=n*2))[:n]
    return f / (f.max() + 1e-6)

def show_status(msg):
    global status_msg, status_timer
    status_msg = msg
    status_timer = FPS * 2

frame_i = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE:
                running = False

            # FIX: Pause/Play Crash 
            elif k == pygame.K_SPACE:
                if paused:
                    paused = False
                    start_ticks = pygame.time.get_ticks()
                    seek_offset = pause_offset
                    # Instead of current_sound.play(start=...), we use our robust slice function
                    restart_playback(pause_offset)
                    show_status("▶ Playing")
                else:
                    paused = True
                    pause_offset = get_pos_samples()
                    current_sound.stop()
                    show_status("⏸ Paused")

            elif k == pygame.K_RIGHT:
                pos = get_pos_samples()
                restart_playback(min(pos + SR * 10, n_samples - 1))
                show_status(">> +10s")
            elif k == pygame.K_LEFT:
                pos = get_pos_samples()
                restart_playback(max(pos - SR * 10, 0))
                show_status("<< -10s")

            elif k == pygame.K_BACKQUOTE:
                show_lyrics = not show_lyrics
                show_status(f"Lyrics {'ON' if show_lyrics else 'OFF'}")

            elif k == pygame.K_RETURN:
                mode = "stem" if mode == "mix" else "mix"
                show_status(f"Mode: {mode.upper()}")

            elif mode == "stem":
                key_map = {
                    pygame.K_1: "other", pygame.K_2: "bass", pygame.K_3: "drums",
                    pygame.K_4: "guitar", pygame.K_5: "vocals", pygame.K_6: "piano",
                }
                if k in key_map:
                    name = key_map[k]
                    stem_enabled[name] = not stem_enabled[name]
                    pos = get_pos_samples()
                    restart_playback(pos)
                    show_status(f"{name}: {'ON' if stem_enabled[name] else 'OFF'}")

            elif mode == "mix":
                mix_map = {
                    pygame.K_z: "z", pygame.K_x: "x", pygame.K_c: "c", pygame.K_v: "v",
                }
                if k in mix_map:
                    preset = MIXES[mix_map[k]]
                    for n in stem_names:
                        stem_enabled[n] = n in preset["on"]
                    pos = get_pos_samples()
                    restart_playback(pos)
                    show_status(f"♪ {preset['name']}")

    pos = get_pos_samples()
    if pos >= n_samples:
        running = False
        break

    t_sec = pos / SR

    # fade for motion blur
    fade = pygame.Surface((W, H), pygame.SRCALPHA)
    fade.fill((0, 0, 0, 50))
    screen.blit(fade, (0, 0))

    # get chunks
    c_other  = get_chunk_mono("other",  pos)
    c_bass   = get_chunk_mono("bass",   pos)
    c_drums  = get_chunk_mono("drums",  pos)
    c_guitar = get_chunk_mono("guitar", pos)
    c_vocals = get_chunk_mono("vocals", pos)

    amp_other  = amplitude(c_other)  * (1 if stem_enabled["other"]  else 0)
    amp_bass   = amplitude(c_bass)   * (1 if stem_enabled["bass"]   else 0)
    amp_drums  = amplitude(c_drums)  * (1 if stem_enabled["drums"]  else 0)
    amp_guitar = amplitude(c_guitar) * (1 if stem_enabled["guitar"] else 0)
    amp_vocals = amplitude(c_vocals) * (1 if stem_enabled["vocals"] else 0)

    bins_bass   = fft_bins(c_bass,   32)
    bins_drums  = fft_bins(c_drums,  16)
    bins_guitar = fft_bins(c_guitar, 6)
    bins_vocals = fft_bins(c_vocals, 36)

    # ── OTHER: floating particles ──────────────────────────────────────
    if stem_enabled["other"]:
        for p in range(N_PARTICLES):
            px[p] += pvx[p] * (1 + amp_other * 3)
            py[p] += pvy[p] * (1 + amp_other * 3)
            if px[p] < 0 or px[p] > W: pvx[p] *= -1
            if py[p] < 0 or py[p] > H: pvy[p] *= -1
            brightness = int(40 + amp_other * 100)
            col = (brightness // 3, brightness // 2, brightness)
            pygame.draw.circle(screen, col, (int(px[p]), int(py[p])), 2)

    # ── BASS: Subwoofer Pulse & Shockwaves ──────────────────────────────
    if stem_enabled["bass"]:
        # 1. The Core Aura (breathes smoothly with the bass volume)
        core_radius = int(80 + amp_bass * 400)
        core_alpha = min(200, int(50 + amp_bass * 500))
        
        if core_radius > 0:
            core_surf = pygame.Surface((core_radius * 2, core_radius * 2), pygame.SRCALPHA)
            # Soft, massive outer glow (Deep Purple)
            pygame.draw.circle(core_surf, (130, 20, 200, core_alpha // 2), (core_radius, core_radius), core_radius)
            # Brighter, tighter inner core (Neon Magenta)
            pygame.draw.circle(core_surf, (220, 50, 255, core_alpha), (core_radius, core_radius), int(core_radius * 0.5))
            screen.blit(core_surf, (W // 2 - core_radius, H // 2 - core_radius))

        # 2. Spawn Shockwaves on heavy hits
        # If the bass spikes up suddenly, trigger a ring
        if amp_bass > prev_bass_amp * 1.3 and amp_bass > 0.06:
            bass_shockwaves.append({
                'r': core_radius, 
                'alpha': 255, 
                'thick': int(10 + amp_bass * 25)
            })
        prev_bass_amp = amp_bass

        # 3. Animate the Shockwaves
        for sw in bass_shockwaves[:]:
            sw['r'] += 14        # Ring expands outward
            sw['alpha'] -= 10    # Ring fades out over time
            
            if sw['alpha'] <= 0:
                bass_shockwaves.remove(sw)
                continue
                
            # Draw the expanding hollow ring
            ring_surf = pygame.Surface((int(sw['r'] * 2), int(sw['r'] * 2)), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (255, 80, 255, sw['alpha']), (int(sw['r']), int(sw['r'])), int(sw['r']), sw['thick'])
            screen.blit(ring_surf, (W // 2 - int(sw['r']), H // 2 - int(sw['r'])))
    else:
        bass_shockwaves.clear()
        prev_bass_amp = 0.0


    # ── DRUMS: Piercing Radial Impacts ─────────────────────────────────
    if stem_enabled["drums"]:
        drum_amp = amplitude(c_drums)
        if drum_amp > prev_drum_amp * 1.4 and drum_amp > 0.05:
            # Spread them slightly further out to avoid crowding the bass core
            ix = np.random.randint(W // 5, 4 * W // 5)
            iy = np.random.randint(H // 5, 4 * H // 5)
            # Increased impact max age and intensity multiplier
            drum_impacts.append([ix, iy, 0, 16, int(drum_amp * 1200)])
        prev_drum_amp = drum_amp

        for imp in drum_impacts[:]:
            imp[2] += 1
            if imp[2] >= imp[3]:
                drum_impacts.remove(imp)
                continue
            progress = imp[2] / imp[3]
            # Larger explosion radius
            radius = int(progress * 130)
            brightness = int(imp[4] * (1 - progress))
            brightness = min(255, brightness)
            n_rays = 12
            
            for r in range(n_rays):
                angle = (r / n_rays) * 2 * math.pi
                ex = int(imp[0] + math.cos(angle) * radius)
                ey = int(imp[1] + math.sin(angle) * radius)
                
                # High-contrast Electric Cyan to cut through the Magenta Bass
                drum_col = (int(brightness * 0.2), brightness, brightness)
                # Thicker strokes (4 instead of 2) for heavy hits
                pygame.draw.line(screen, drum_col, (imp[0], imp[1]), (ex, ey), 4)
    else:
        drum_impacts.clear()

    # ── GUITAR: 6 diagonal strings beyond frame ────────────────────────
    if stem_enabled["guitar"]:
        for s_i, (x1, y1, x2, y2) in enumerate(STRINGS):
            val = bins_guitar[s_i] * amp_guitar
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length == 0:
                continue
            px_perp = -dy / length
            py_perp = dx / length
            s_pts = []
            
            for k in range(60):
                tt = k / 59
                vib = math.sin(tt * math.pi * 4 + frame_i * 0.4) * val * 40
                s_pts.append((
                    int(x1 + tt * dx + px_perp * vib),
                    int(y1 + tt * dy + py_perp * vib)
                ))
            
            brightness = min(255, int(150 + val * 400))
            col = (int(brightness * 0.8), int(brightness * 0.95), brightness)
            
            # 2. String Gauges: 
            string_thickness = max(1, 7 - s_i)
            
            if len(s_pts) > 1:
                pygame.draw.lines(screen, col, False, s_pts, string_thickness)

    # ── VOCALS: radial lines from center ──────────────────────────────
    if stem_enabled["vocals"]:
        rotation = frame_i * 0.02
        cx, cy = W // 2, H // 2
        for r_i in range(36):
            angle = (r_i / 36) * 2 * math.pi + rotation
            val = bins_vocals[r_i] * amp_vocals
            line_len = int(20 + val * 150)
            ex = int(cx + math.cos(angle) * line_len)
            ey = int(cy + math.sin(angle) * line_len)
            brightness = min(255, int(80 + val * 500))
            pygame.draw.line(screen, (brightness, brightness, brightness),
                             (cx, cy), (ex, ey), 1)

    # ── LYRICS ────────────────────────────────────────────────────────
    if show_lyrics:
        sub_text = get_subtitle(t_sec)
        if sub_text:
            active_font = font_jap if has_japanese(sub_text) else font_eng

            # 1. Calculate Intensity from Vocals
            # We subtract a tiny baseline (0.02) so quiet singing stays crisp and readable
            intensity = max(0.0, amp_vocals - 0.02) 
            
            # Scale the glitch and shake based on how loud the singer is
            aberration = int(intensity * 30) 
            
            shake_x = 0
            shake_y = 0
            if intensity > 0.05: # Only shake on loud notes/screams
                shake_x = np.random.randint(-int(intensity * 30), int(intensity * 30) + 1)
                shake_y = np.random.randint(-int(intensity * 30), int(intensity * 30) + 1)

            # 2. Render the layers
            surfs_white, line_height = render_text_wrapped(sub_text, active_font, (255, 255, 255), W - 100)
            surfs_red, _ = render_text_wrapped(sub_text, active_font, (255, 20, 20), W - 100)
            surfs_cyan, _ = render_text_wrapped(sub_text, active_font, (20, 255, 255), W - 100)
            shadows, _ = render_text_wrapped(sub_text, active_font, (0, 0, 0), W - 100)
            
            total_height = len(surfs_white) * line_height
            start_y = H - total_height - 60 
            
            for i in range(len(surfs_white)):
                base_x = W // 2 - surfs_white[i].get_width() // 2 + shake_x
                y = start_y + i * line_height + shake_y
                
                # Draw thick shadow for readability
                screen.blit(shadows[i], (base_x + 3, y + 3))
                
                if aberration > 0:
                    # Draw Chromatic Fringes
                    screen.blit(surfs_red[i], (base_x, y - aberration))
                    screen.blit(surfs_cyan[i], (base_x + aberration, y))
                
                # Draw Core White Text on top
                screen.blit(surfs_white[i], (base_x, y))

    # ── HUD ───────────────────────────────────────────────────────────
    t_str = f"{int(t_sec//60):02d}:{int(t_sec%60):02d} / {int(n_samples/SR//60):02d}:{int(n_samples/SR%60):02d}"
    screen.blit(font.render(t_str, True, (180, 180, 180)), (10, 10))

    mode_col = (100, 255, 100) if mode == "stem" else (100, 100, 255)
    screen.blit(font.render(f"[ENTER] Mode: {mode.upper()}", True, mode_col), (10, 32))

    if mode == "stem":
        for i, name in enumerate(stem_names):
            col = (100, 255, 100) if stem_enabled[name] else (100, 100, 100)
            screen.blit(font.render(f"{i+1}:{name}", True, col), (10, 54 + i * 20))

    if status_timer > 0:
        alpha = min(255, status_timer * 8)
        s = font_large.render(status_msg, True, (255, 255, 100))
        screen.blit(s, (W // 2 - s.get_width() // 2, H // 2 - 20))
        status_timer -= 1

    if paused:
        screen.blit(font_large.render("⏸ PAUSED", True, (255, 200, 0)), (W - 160, 10))

    pygame.display.flip()
    frame_i += 1
    clock.tick(FPS)

pygame.quit()
print("Done")
