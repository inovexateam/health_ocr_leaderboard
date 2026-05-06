# Health OCR Leaderboard

WhatsApp group fitness leaderboard powered by 100% open-source OCR.
No paid APIs. No internet required after setup.

## Tech stack
- **OCR**: Tesseract 5 + OpenCV (image preprocessing)
- **Backend**: Python / Flask + SQLite
- **Frontend**: Vanilla HTML/CSS/JS (single file, no build step)

## Requirements
- Python 3.8+
- Tesseract 5 (`sudo apt install tesseract-ocr`)

## Install

```bash
pip install flask pytesseract pillow opencv-python
```

## Run

```bash
python app.py
# Open http://localhost:5050
```

## What it does

1. **Test OCR** tab — pick any of 5 bundled sample images (Apple Health,
   Google Fit, Fitbit, Samsung Health, manual WhatsApp post) and see
   exactly what the OCR extracts.

2. **Upload Image** tab — upload any real health app screenshot, enter the
   member name, and save it to the leaderboard.

3. **Leaderboard** tab — ranked view with podium, bar charts, daily/weekly/
   monthly periods, and multiple sort metrics (steps, calories, distance,
   active minutes).

4. **All Entries** tab — raw log of every submission.

## Supported formats

The OCR engine handles:
- Apple Health screenshots
- Google Fit screenshots  
- Fitbit screenshots
- Samsung Health screenshots
- Garmin / Strava screenshots
- Manual WhatsApp text posts (typed stats)
- Any image with numeric fitness data

## How the OCR works

Each image goes through 6 preprocessing variants:
1. Plain grayscale
2. Otsu threshold (good for dark-bg apps)
3. Inverted Otsu (white-on-dark text)
4. Adaptive threshold (uneven lighting)
5. CLAHE + denoising (compressed screenshots)
6. Sharpening

All variants are OCR'd by Tesseract and the best combined text is parsed
with regex patterns for each fitness metric.

## WhatsApp integration (next step)

Replace the upload UI with a whatsapp-web.js bot:

```js
const { Client, LocalAuth } = require('whatsapp-web.js');
const client = new Client({ authStrategy: new LocalAuth() });

client.on('message', async (msg) => {
  if (msg.from === 'GROUPID@g.us' && msg.hasMedia) {
    const media = await msg.downloadMedia();
    const buf = Buffer.from(media.data, 'base64');
    // POST buf to /api/process with member name from msg.author
  }
});
```
