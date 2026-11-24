"""Draw Passage's app icon in the arrow-poster style: an indigo arrow over a lime
pinstripe band and a black arrowhead, all pointing up-right on a light tile.

Writes passage.ico (16-256 px) and passage-icon.png (512 px). Front-end asset only.
"""
import math
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent
S = 1024
PAPER, INDIGO, LIME, LIME_DARK = (243, 243, 246, 255), (47, 51, 198, 255), (196, 200, 42, 255), (154, 157, 23, 255)
INK, GRAY = (18, 18, 22, 255), (170, 170, 180, 255)

def arrow(tip, length, width, head):
    """Square-cut 90-degree arrow pointing up-right with its tip at `tip`."""
    d = (math.sqrt(.5), -math.sqrt(.5)); n = (math.sqrt(.5), math.sqrt(.5)); t = head / 2
    local = [(-length, -width / 2), (-t, -width / 2), (-t, -head / 2), (0, 0), (-t, head / 2), (-t, width / 2), (-length, width / 2)]
    return [(tip[0] + u * d[0] + v * n[0], tip[1] + u * d[1] + v * n[1]) for u, v in local]

def draw():
    art = Image.new('RGBA', (S, S), PAPER); d = ImageDraw.Draw(art)
    d.polygon(arrow((900, 300), 1100, 120, 250), fill=GRAY)
    lime = arrow((640, 300), 1100, 150, 150); d.polygon(lime, fill=LIME)
    inside = Image.new('L', (S, S)); ImageDraw.Draw(inside).polygon(lime, fill=255)
    lines = Image.new('RGBA', (S, S), (0, 0, 0, 0)); ld = ImageDraw.Draw(lines)
    for x in range(0, S, 22): ld.line([(x, 0), (x, S)], fill=LIME_DARK, width=6)
    lines.putalpha(ImageChops.multiply(lines.getchannel('A'), inside)); art.alpha_composite(lines)
    d = ImageDraw.Draw(art)
    d.polygon(arrow((820, 200), 1200, 230, 520), fill=INDIGO)
    d.polygon(arrow((470, 560), 700, 120, 270), fill=INK)
    tile = Image.new('L', (S, S)); ImageDraw.Draw(tile).rounded_rectangle((24, 24, S - 24, S - 24), radius=200, fill=255)
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); im.paste(art, (0, 0), tile)
    return im

def main():
    im = draw()
    im.resize((512, 512), Image.Resampling.LANCZOS).save(ROOT / 'passage-icon.png')
    im.save(ROOT / 'passage.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
    print('icon written')

if __name__ == '__main__':
    main()
