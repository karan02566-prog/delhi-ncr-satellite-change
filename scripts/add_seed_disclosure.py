from pathlib import Path

p = Path("app/app.py")
text = p.read_text(encoding="utf-8")

anchor = "A rectangular near-zero region in the built-up mask (Built-up Mask tab) reflects a genuinely low-built-up area, confirmed via 100% valid-pixel coverage in the underlying patches (no missing/nodata pixels), ruling out a data-coverage artifact."
addition = "\n- Training runs did not fix a random seed; reported metrics reflect one training run and are not guaranteed bit-for-bit reproducible across reruns, though data/split/architecture/hyperparameters were held constant throughout."

if anchor not in text:
    print("ANCHOR NOT FOUND")
else:
    text = text.replace(anchor, anchor + addition)
    p.write_text(text, encoding="utf-8")
    print("Added seed disclosure bullet.")
