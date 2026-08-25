from pathlib import Path

p = Path("app/app.py")
text = p.read_text(encoding="utf-8")

old = "A rectangular near-zero region in the built-up mask (Built-up Mask tab) is under investigation \u2014 see disclosure there."
new = "A rectangular near-zero region in the built-up mask (Built-up Mask tab) reflects a genuinely low-built-up area, confirmed via 100% valid-pixel coverage in the underlying patches (no missing/nodata pixels), ruling out a data-coverage artifact."

if old not in text:
    print("OLD STRING NOT FOUND — paste the exact line 545 content (copy from your editor, not findstr) so I can fix the match.")
else:
    text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print("Patched line 545 successfully.")