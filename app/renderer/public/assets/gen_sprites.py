"""Generate pixel-art character sprites matching the classroom_bg.png art style.

Creates individual character PNGs and a combined spritesheet for Phaser.
Each character: 40x56px, 2 frames (idle + walk), facing down (3/4 top-down view).
Style: chibi proportions, large head, small body — matches the classroom background.
"""
from PIL import Image, ImageDraw

# ── Color palette (matched to classroom_bg.png) ─────────────────────────
SKIN = (240, 200, 160)
SKIN_SHADOW = (220, 175, 135)
EYE = (40, 35, 45)
MOUTH = (200, 130, 110)
BLUSH = (255, 180, 170, 80)
PANTS = (50, 55, 70)
SHOE = (60, 45, 35)
OUTLINE = (60, 55, 80)
WHITE = (255, 255, 255)

# Character definitions: (hair_color, shirt_color, hair_style)
# hair_style: 0=short, 1=medium, 2=long, 3=spiky
TEACHER = {
    "hair": (85, 70, 50),
    "shirt": (240, 240, 235),  # white shirt
    "style": 1,
    "glasses": True,
    "tie": (190, 50, 50),
}

STUDENTS = [
    {"hair": (85, 50, 30),   "shirt": (70, 120, 170),  "style": 0},  # S01
    {"hair": (135, 100, 60), "shirt": (70, 170, 120),  "style": 1},  # S02
    {"hair": (50, 35, 70),   "shirt": (170, 100, 70),  "style": 3},  # S03
    {"hair": (35, 20, 35),   "shirt": (85, 135, 185),  "style": 0},  # S04
    {"hair": (100, 65, 35),  "shirt": (100, 170, 85),  "style": 2},  # S05
    {"hair": (70, 50, 35),   "shirt": (185, 85, 85),   "style": 1},  # S06
    {"hair": (85, 70, 50),   "shirt": (120, 100, 170), "style": 3},  # S07
    {"hair": (50, 35, 20),   "shirt": (70, 185, 170),  "style": 0},  # S08
    {"hair": (100, 85, 65),  "shirt": (170, 120, 70),  "style": 2},  # S09
    {"hair": (35, 35, 50),   "shirt": (85, 155, 170),  "style": 1},  # S10
    {"hair": (120, 100, 85), "shirt": (70, 135, 100),  "style": 0},  # S11
    {"hair": (70, 50, 85),   "shirt": (185, 120, 100), "style": 3},  # S12
    {"hair": (85, 70, 35),   "shirt": (100, 135, 185), "style": 2},  # S13
    {"hair": (50, 35, 50),   "shirt": (120, 170, 85),  "style": 1},  # S14
    {"hair": (100, 85, 50),  "shirt": (170, 85, 120),  "style": 0},  # S15
    {"hair": (70, 50, 20),   "shirt": (85, 170, 170),  "style": 2},  # S16
    {"hair": (35, 20, 50),   "shirt": (135, 185, 85),  "style": 3},  # S17
    {"hair": (85, 70, 20),   "shirt": (185, 135, 70),  "style": 1},  # S18
    {"hair": (50, 50, 70),   "shirt": (85, 120, 185),  "style": 0},  # S19
    {"hair": (100, 65, 50),  "shirt": (70, 170, 135),  "style": 2},  # S20
]

CHAR_W, CHAR_H = 40, 56

def _darker(color, factor=0.75):
    return tuple(int(c * factor) for c in color[:3])

def _lighter(color, factor=1.25):
    return tuple(min(255, int(c * factor)) for c in color[:3])

def draw_character(draw, ox, oy, char, frame=0, is_teacher=False):
    """Draw a single chibi character at (ox, oy) top-left."""
    hair = char["hair"]
    shirt = char["shirt"]
    style = char["style"]
    hair_dark = _darker(hair)
    shirt_dark = _darker(shirt)
    shirt_light = _lighter(shirt)

    # ── Shadow on ground ──
    for dx in range(-8, 9):
        for dy in range(-2, 3):
            if dx*dx/81 + dy*dy/9 <= 1:
                draw.point((ox + 20 + dx, oy + 53 + dy), fill=(0, 0, 0, 50))

    # ── Legs + Shoes ──
    if frame == 0:
        # Idle: legs together-ish
        draw.rectangle([ox+13, oy+42, ox+17, oy+50], fill=PANTS)
        draw.rectangle([ox+22, oy+42, ox+26, oy+50], fill=PANTS)
        draw.rectangle([ox+12, oy+50, ox+18, oy+53], fill=SHOE)
        draw.rectangle([ox+21, oy+50, ox+27, oy+53], fill=SHOE)
    else:
        # Walk: legs apart
        draw.rectangle([ox+11, oy+42, ox+16, oy+51], fill=PANTS)
        draw.rectangle([ox+23, oy+42, ox+28, oy+49], fill=PANTS)
        draw.rectangle([ox+10, oy+51, ox+17, oy+53], fill=SHOE)
        draw.rectangle([ox+22, oy+49, ox+29, oy+52], fill=SHOE)

    # ── Body / Torso ──
    draw.rectangle([ox+10, oy+30, ox+29, oy+43], fill=shirt)
    # Shirt shading
    draw.rectangle([ox+10, oy+30, ox+13, oy+43], fill=shirt_dark)
    draw.rectangle([ox+26, oy+30, ox+29, oy+43], fill=shirt_dark)
    # Collar
    draw.rectangle([ox+15, oy+29, ox+24, oy+32], fill=shirt_light)

    # ── Arms ──
    if frame == 0:
        draw.rectangle([ox+6, oy+32, ox+10, oy+41], fill=shirt)
        draw.rectangle([ox+29, oy+32, ox+33, oy+41], fill=shirt)
        draw.rectangle([ox+6, oy+40, ox+10, oy+43], fill=SKIN)
        draw.rectangle([ox+29, oy+40, ox+33, oy+43], fill=SKIN)
    else:
        draw.rectangle([ox+5, oy+31, ox+10, oy+39], fill=shirt)
        draw.rectangle([ox+29, oy+33, ox+34, oy+42], fill=shirt)
        draw.rectangle([ox+5, oy+38, ox+10, oy+41], fill=SKIN)
        draw.rectangle([ox+29, oy+41, ox+34, oy+44], fill=SKIN)

    # ── Head (large, chibi proportion) ──
    # Main head shape
    draw.rectangle([ox+8, oy+6, ox+31, oy+30], fill=SKIN)
    # Rounded top
    draw.rectangle([ox+10, oy+4, ox+29, oy+7], fill=SKIN)
    # Rounded bottom (chin)
    draw.rectangle([ox+12, oy+29, ox+27, oy+31], fill=SKIN)
    # Ear hints
    draw.rectangle([ox+7, oy+15, ox+9, oy+22], fill=SKIN_SHADOW)
    draw.rectangle([ox+30, oy+15, ox+32, oy+22], fill=SKIN_SHADOW)

    # ── Hair ──
    # Top hair (all styles)
    draw.rectangle([ox+7, oy+2, ox+32, oy+10], fill=hair)
    draw.rectangle([ox+9, oy+0, ox+30, oy+4], fill=hair)

    if style == 0:  # short
        draw.rectangle([ox+7, oy+2, ox+10, oy+16], fill=hair)
        draw.rectangle([ox+29, oy+2, ox+32, oy+16], fill=hair)
        # Bangs
        draw.rectangle([ox+10, oy+6, ox+15, oy+12], fill=hair)
        draw.rectangle([ox+24, oy+6, ox+29, oy+12], fill=hair)
    elif style == 1:  # medium
        draw.rectangle([ox+7, oy+2, ox+11, oy+20], fill=hair)
        draw.rectangle([ox+28, oy+2, ox+32, oy+20], fill=hair)
        # Bangs
        draw.rectangle([ox+10, oy+6, ox+16, oy+13], fill=hair)
        draw.rectangle([ox+23, oy+6, ox+29, oy+13], fill=hair)
        draw.rectangle([ox+15, oy+6, ox+24, oy+10], fill=hair)
    elif style == 2:  # long
        draw.rectangle([ox+6, oy+2, ox+11, oy+28], fill=hair)
        draw.rectangle([ox+28, oy+2, ox+33, oy+28], fill=hair)
        # Bangs
        draw.rectangle([ox+10, oy+6, ox+16, oy+14], fill=hair)
        draw.rectangle([ox+23, oy+6, ox+29, oy+14], fill=hair)
        draw.rectangle([ox+15, oy+6, ox+24, oy+10], fill=hair)
    elif style == 3:  # spiky
        draw.rectangle([ox+6, oy+2, ox+10, oy+15], fill=hair)
        draw.rectangle([ox+29, oy+2, ox+33, oy+15], fill=hair)
        # Spikes on top
        draw.rectangle([ox+8, oy+0, ox+12, oy+3], fill=hair_dark)
        draw.rectangle([ox+15, oy+0, ox+19, oy+2], fill=hair)
        draw.rectangle([ox+22, oy+0, ox+26, oy+2], fill=hair)
        draw.rectangle([ox+28, oy+0, ox+32, oy+3], fill=hair_dark)
        # Bangs
        draw.rectangle([ox+11, oy+6, ox+15, oy+12], fill=hair)
        draw.rectangle([ox+24, oy+6, ox+28, oy+12], fill=hair)

    # Hair highlight
    draw.rectangle([ox+14, oy+3, ox+18, oy+5], fill=_lighter(hair, 1.3))

    # ── Face ──
    # Eyes (white + pupil)
    draw.rectangle([ox+13, oy+16, ox+17, oy+20], fill=WHITE)
    draw.rectangle([ox+22, oy+16, ox+26, oy+20], fill=WHITE)
    draw.rectangle([ox+14, oy+17, ox+16, oy+19], fill=EYE)
    draw.rectangle([ox+23, oy+17, ox+25, oy+19], fill=EYE)
    # Eye shine
    draw.point((ox+14, oy+17), fill=WHITE)
    draw.point((ox+23, oy+17), fill=WHITE)

    # Mouth
    draw.rectangle([ox+18, oy+23, ox+21, oy+24], fill=MOUTH)

    # Blush
    draw.rectangle([ox+11, oy+21, ox+14, oy+23], fill=(255, 180, 170, 60))
    draw.rectangle([ox+25, oy+21, ox+28, oy+23], fill=(255, 180, 170, 60))

    # ── Teacher extras ──
    if is_teacher:
        # Glasses
        draw.rectangle([ox+12, oy+15, ox+18, oy+21], outline=(80, 80, 80))
        draw.rectangle([ox+21, oy+15, ox+27, oy+21], outline=(80, 80, 80))
        draw.line([(ox+18, oy+17), (ox+21, oy+17)], fill=(80, 80, 80))
        # Tie
        tie_color = char.get("tie", (190, 50, 50))
        draw.polygon([(ox+18, oy+32), (ox+21, oy+32), (ox+20, oy+40), (ox+19, oy+40)], fill=tie_color)


def generate_sprites():
    # ── Generate individual character PNGs ──
    all_chars = [("teacher", TEACHER, True)] + [
        (f"S{i+1:02d}", s, False) for i, s in enumerate(STUDENTS)
    ]

    # Spritesheet: each character gets 2 frames (idle + walk) side by side
    # Layout: 21 rows × 2 columns, each cell 40×56
    sheet_w = CHAR_W * 2
    sheet_h = CHAR_H * len(all_chars)
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    sheet_draw = ImageDraw.Draw(sheet)

    for row, (name, char, is_teacher) in enumerate(all_chars):
        # Frame 0 (idle)
        draw_character(sheet_draw, 0, row * CHAR_H, char, frame=0, is_teacher=is_teacher)
        # Frame 1 (walk)
        draw_character(sheet_draw, CHAR_W, row * CHAR_H, char, frame=1, is_teacher=is_teacher)

        # Also save individual idle frame
        single = Image.new("RGBA", (CHAR_W, CHAR_H), (0, 0, 0, 0))
        single_draw = ImageDraw.Draw(single)
        draw_character(single_draw, 0, 0, char, frame=0, is_teacher=is_teacher)
        single.save(f"char_{name}.png")
        print(f"Generated char_{name}.png")

    sheet.save("spritesheet.png")
    print(f"\nSpritesheet saved: spritesheet.png ({sheet_w}x{sheet_h})")
    print(f"  {len(all_chars)} characters × 2 frames")
    print(f"  Frame size: {CHAR_W}x{CHAR_H}")
    print(f"  Row order: teacher, S01..S20")


if __name__ == "__main__":
    generate_sprites()
