# Enhance Speech

A local, private rebuild of the Adobe Podcast "Enhance Speech" tool. Drop in audio or video and get speech that sounds studio-recorded, with live sliders for speech, music and background. Everything runs on your own computer.

## Run

```bash
./run.sh
```

Then open http://localhost:5840. The first run installs Python packages. The first file you process downloads the AI models (about 2 GB, and the BandIt v2 file comes from Zenodo, which can be slow), and the first transcript fetches Whisper. It needs `ffmpeg` and `uv`, both from Homebrew.

## What it does

- Drag and drop, or choose, one file or many. They process in a queue.
- Audio: wav, mp3, m4a, aac, flac, ogg. Video: mp4, mov, m4v and more.
- Speech, Music and Background sliders change the mix instantly while it plays.
- Enhance Speech switch compares against the original at any moment.
- Download the mix, or the speech, music or background stem on its own. Video keeps its original picture stream untouched when the output container can hold its codec; otherwise the picture is re-encoded to H.264.
- Transcript (txt) and captions (srt), made locally with Whisper.
- v2 is the full restoration. v1 is a quicker clean-up that blends with the original.

## How it works

| Step | Model or method | Why |
|---|---|---|
| Very quiet recordings | The whole chain runs at a normal level, and the stems come back down at the end (`server/pipeline.py`) | DeepFilterNet removes speech under about -40 dBFS as if it were silence |
| Separate speech, music, effects | BandIt v2 (multilingual) | Closest match to Adobe's stems when scored against them |
| Rescue speech the separator dropped, clear leftover noise | DeepFilterNet3, Silero VAD | Heavy noise and echo can make the separator file the voice as "effects" |
| Regenerate the speech | Sidon, fed the speech before the second denoiser pass | Gives the studio envelope and the sound above 8 kHz |
| Keep the voice itself | The recording's waveform shaped by Sidon's envelope below 4-8 kHz (`server/dsp.py`) | Adobe's output stays locked to the recording below about 2 kHz; ours goes higher on purpose, so the pitch stays yours and slider blends do not comb-filter. Words hold on normal recordings. Where noise buried the waveform, Sidon's own sound is used, and there words can change |
| Harmonic clarity | A pitch-following comb, as deep as the file falls short of a clean voice's harmonics (`server/clarity.py`) | Adobe's harmonics stand clear of the room on every voice measured. A clear recording is left alone |
| Tone | One filter per file (`server/tone.py`) | Each voice keeps its own spectrum. The filter is a rumble cut and a low balance set to what Adobe's outputs share |
| Fricative brightness | Measured from the recording, once per file (`server/highs.py`) | Sidon only hears up to 8 kHz, so its "s" and "sh" highs are a guess |
| Limit by the recording | From 1 kHz up, the stem's band envelope may not rise above the recording's (`server/limit.py`) | Regenerated speech puts energy where the recording has none |
| Gaps between words | Speech gate, a low-band gate and a faint noise floor (`server/gate.py`) | Near-silent gaps that open before a word and leave quiet speech alone. Adobe's gaps hold a floor about 72 dB under the speech rather than digital black. The speech gate backs off when the separator lost the voice and the speech stem holds more than 6 dB less than the input while someone speaks, and is off below -10 dB, because under that much noise what is quiet is often the words. Loud music alone does not count for it: the separator handles music well. The low-band gate also fades out as the speech stem falls from 3 to 8 dB under the input over the whole file, music included. The floor always stays |
| Loudness | -20.8 LUFS on the enhanced stem (`server/level.py`) | Adobe's stem sits at -21.2, -20.9 and -20.4 LUFS on three very different inputs. A recording with no voice is not lifted |
| Loud moments | Left out of every file-wide measure, then brought down to about the talker's own loudest frames (`server/gate.py`) | A cough or a word close to the mic would otherwise set the file's level and tone, and leave the talker far too quiet |

The player streams one MP4 holding all stems as 8 audio channels (the same layout Adobe uses) and mixes them in the browser with Web Audio, so sliders respond with no re-processing. `server/dsp.py` and `web/js/player.js` share the same gain formula. A v2 download then brings the mono speech stems down 3 dB, as Adobe's downloads do (measured as 0.7071 on two of its full mixes, one of them with music and background muted; the Speech card's own download is inferred from those), while the player leaves them alone as Adobe's player does. No Adobe v1 file was available, so v1 downloads stay at unity.

Transcripts read the enhanced stem. Under noise or distortion so heavy that the separator lost the voice, the regenerated speech can drop words the raw recording still has, so Whisper reads both and keeps the one it is more confident of.

## Project layout

- `server/` FastAPI app, job queue, pipeline, model wrappers
- `server/vendor/bandit_v2/` BandIt v2 model code (MIT, from Music-Source-Separation-Training), trimmed to inference only
- `web/` the interface, plain HTML, CSS and JavaScript with no build step
- `bench/` the scripts that chose the models by scoring them against reference recordings
- `data/` your uploads and results; delete a track in the app to remove its files
- `models/` BandIt and MossFormer checkpoints. Sidon and Whisper go to the Hugging Face cache (`~/.cache/huggingface`), DeepFilterNet to `~/Library/Caches`

## Licences to check before any public or commercial use

- This is a functional and visual rebuild. It uses no Adobe code, fonts, icons, audio or images. Adobe's name and logo are left out, and the wording is rewritten where it referred to Adobe products or plans.
- Model weights: BandIt v2 is CC BY-SA 4.0 (credit the authors; changes to the model itself must be shared alike). Sidon v0.1 and w2v-BERT 2.0 are MIT. MossFormer2 is Apache 2.0. DeepFilterNet is MIT or Apache 2.0. Whisper and Silero VAD are MIT.
- Source Sans 3 is SIL Open Font License.
- Earlier BandIt weights (BandIt Plus) were tested and dropped because they are non-commercial (CC BY-NC 4.0).
- The landing page sample is a public-domain LibriVox reading of *The Art of War*, with room echo and fan noise added for the demo.
