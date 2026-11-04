# Bench

How the chain was chosen and how close it gets. Adobe's real output exists for three voices here, so every choice was judged on all three:

- **The 1102 clip:** a no-mic room recording, pitch about 160 Hz, with Adobe's download at speech 100%, music and background muted.
- **Audio.MP3:** a second real recording with a dead right channel, with Adobe's download at speech 100%, background muted, music 26%.
- **Adobe's sample:** speech and music outdoors, pitch about 246 Hz. Its public sample video holds all stems as an 8-channel FLAC track. Its highs carry codec artefacts.

A curve fitted on one voice and scored on that voice proves nothing. The first version of this chain did exactly that with its tone target, and that was the largest error found.

## Against Adobe's output

From `scorecard.py`, enhanced stem against Adobe's. Lower distance and tone error are closer. The other rows are closest when they match the Adobe column.

| Measure | 1102: old | now | Adobe | Audio: old | now | Adobe | Sample: old | now | Adobe |
|---|---|---|---|---|---|---|---|---|---|
| Log-mel distance | 6.68 | 4.56 | | 6.38 | 4.21 | | 5.70 | 5.76 | |
| Vowel tone error, dB | 5.7 | 1.5 | | 4.7 | 2.2 | | 1.5 | 2.3 | |
| Fricative tone error, dB | 7.2 | 1.5 | | 4.8 | 2.5 | | 2.3 | 2.6 | |
| Pitch against Adobe's, cents | 12.1 | 6.1 | | 11.6 | 5.0 | | 10.8 | 7.3 | |
| HNR | 10.8 | 12.4 | 13.0 | 10.9 | 13.6 | 14.2 | 14.3 | 16.1 | 14.6 |
| CPPS | 8.17 | 9.56 | 9.77 | 10.05 | 10.99 | 11.19 | 8.66 | 9.58 | 11.57 |
| Harmonic contrast below 1 kHz, dB | 24.4 | 27.4 | 28.5 | 24.2 | 27.5 | 28.1 | 32.0 | 34.2 | 32.5 |
| Gap distance | 13.10 | 9.40 | | 9.04 | 6.48 | | 13.81 | 9.25 | |
| Waveform match, 200 Hz-1 kHz | 0.89 | 0.95 | | 0.87 | 0.94 | | 0.77 | 0.80 | |
| Waveform match, 1-2 kHz | 0.75 | 0.89 | | 0.74 | 0.89 | | 0.72 | 0.83 | |
| Waveform match, 2-4 kHz | 0.50 | 0.68 | | 0.54 | 0.70 | | 0.44 | 0.48 | |
| Time below -40 dB, % | | 15.2 | 16.4 | | 7.8 | 7.5 | | 22.3 | 23.4 |
| Time below -50 dB, % | | 12.3 | 14.3 | | 6.3 | 5.5 | | 20.1 | 21.2 |
| Time below -60 dB, % | | 10.3 | 10.9 | | 5.1 | 3.7 | | 18.0 | 13.6 |
| Stem loudness, LUFS | | -20.8 | -21.2 | | -20.8 | -20.9 | | -20.8 | -20.4 |

The sample's distance and tone errors end slightly worse than the old chain's, which was fitted on that voice. Everything else on it improves or stays close.

How much of the distance is left to win, for 1102, Audio and the sample: the raw recording scores 9.96, 9.35 and 11.06. This chain scores 4.56, 4.21 and 5.76. The same structure guided by Adobe's own envelope scores 1.71, 1.67 and 2.18. So the blend is the right shape, and most of what remains is Sidon's envelope against Adobe's private model. A literal match is out of reach without that model.

Word match against the original's transcript (Whisper large-v3), enhanced then Adobe: 1102 0.989 and 0.983, Audio 0.980 and 0.963, sample 0.996 and 0.996. A brand name in the recording survives in enhanced and in Adobe's. The isolated stem garbles it into one wrong word, which is why Sidon hears the speech before the second denoiser pass.

DNSMOS cannot see these differences (Adobe 3.36, the old chain 3.31 on 1102), so it was not used to choose.

## What Adobe's own files showed

- **Loudness:** the enhanced stem sits at -21.2, -20.9 and -20.4 LUFS on inputs from -13.8 to -24.5 LUFS, so the target is fixed; -20.8 is the midpoint.
- **Download gain:** a download scales the mono speech stems by 0.7071. The 1102 download has the histogram comb a 0.7071 gain leaves on 16-bit values, and a linear fit of Adobe's preview MP3 gives enhanced and isolated 0.343 (0.5 x 0.7071 x 0.97) with music and background at their plain slider gains. Its browser player connects the mono stems to both channels at unity, so a download sits 3 dB lower on speech than the player.
- **Waveform:** Adobe's output stays locked to the recording below about 2 kHz (correlation 0.93 and 0.82 at 200 Hz-1 kHz, 0.88 and 0.83 at 1-2 kHz). Regenerating the waveform moved pitch by twice as much as Adobe does.
- **Harmonic contrast:** per harmonic number it is nearly one curve across all three voices, which sets the target of `server/clarity.py`.
- **Tone:** Adobe's spectrum differs by up to 32 dB in bass and 15 dB at 10 kHz between voices, which rules out a target curve. One thing agrees: the octave around the pitch against 350-700 Hz, about 2.2 dB, and that sets the low balance.
- **Fricatives:** at 8-12.5 kHz, relative to vowels, Adobe follows the 1102 recording within 0.3 dB, while Sidon guesses 8 dB low. On the sample, whose highs are codec artefacts, Adobe does not follow them, hence the replication guard.
- **Gaps:** they hold a near-white floor about 72 dB under the speech, not digital black.
- **Sidon's delay:** 360 samples, from word onsets on both voices and at every offset of its frame grid. Measuring it per file gave -240 to +96 samples under heavy noise and scored no better on clean files.

## Loud moments and quiet recordings

Each clip is 30 s of a real voice with one sound added close to the mic, or the whole clip turned down. The real chain ran on each, and the figure is the talker's loudness in the enhanced stem, measured away from the sound. `loud_moment_check.py` repeats this.

| Clip | Same clip without the sound, LUFS | Before the fix | Now |
|---|---|---|---|
| 1102 voice, a word 20 dB over the speech | -20.82 | -24.43 | -21.55 |
| 1102 voice, a word 30 dB over the speech | -20.82 | -28.06 | -21.38 |
| 1102 voice, a cough-like burst 30 dB over the speech | -20.86 | silent | -20.96 |
| Audio voice, a word 30 dB over the speech | -20.62 | -31.44 | -20.99 |
| Sample voice, a word 30 dB over the speech | -20.90 | -38.02 | -21.16 |
| 1102 voice, whole clip 33 dB down | -20.89 | not run | -20.90 |
| 1102 voice, whole clip 60 dB down | -20.89 | not run | -20.90 |

The talker's tone away from the sound moved by up to 11 dB before and now moves by at most 2.3 dB (12-16 kHz on one clip), and usually under 1 dB. Away from the sound, the talker's waveform matches the clean run at 16 to 27 dB SI-SDR, and 29 dB on the turned-down clips. The word now lands about 7 dB over the talker's 95th percentile, and the limiter takes under 3 dB off its peaks. The quick v1 chain reads -20.65 to -20.82 on the same kinds of clip. Clips with no loud moment come out bit-identical to the code before this change. A quiet room with no voice at -60 dB stays silent. A near talker who speaks in turns, 30% of a clip at 15 dB over the main talker, came out 7 dB over them, with the main talker at -24.0 LUFS.

What was going wrong, in the order it mattered:

- **The denoiser:** DeepFilterNet removed speech whose 95th percentile sat under about -40 to -44 dBFS, by 30 to 70 dB. A loud moment that sets the recording's peak puts a distant talker there. So does a very quiet recording with no loud moment at all.
- **`level.follow_input`:** Sidon brings a loud moment out 9 to 16 dB quieter against the speech than it went in, so the moment's energy lifted the speech by up to 12 dB in the blocks around it. `limit.py` then took that back from 1 kHz up, which is what moved the tone.
- **File-wide readings:** the loudness, the gate's reference, the floor, the low band's reference, the tone and the highs all read a power mean or a percentile that one loud moment could set.

## Results across recordings

Whisper small word match against the original's transcript, or against clean speech where one exists. DNSMOS for reference. Mixes use the player's gains; a v2 download has speech 3 dB lower against music and background.

Only this first row was measured on the final chain:

| Recording | Original DNSMOS | Speech 100%, rest off | Default 50/10/10 | Word match: original, 100%, default |
|---|---|---|---|---|
| Audio.MP3 | 2.91 | 3.13 | 3.18 | 1.000, 0.963, 0.978 |

These rows predate the clarity, limit, low gate and floor steps, and `final_scores.py` was not rerun on them:

| Recording | Original DNSMOS | Speech 100%, rest off | Default 50/10/10 | Word match: original, 100%, default |
|---|---|---|---|---|
| Adobe's sample | 1.24 | 3.16 | 3.22 | 1.000, 0.992, 0.992 |
| 1102 | 1.97 | 3.22 | 3.10 | 1.000, 0.974, 0.972 |
| 1102 under music, speech 5 dB above it | 1.43 | 3.24 | 3.13 | 1.000, 0.979, 0.979 |
| Echo 0.9 s, fan at 15 dB SNR, MP3 | 1.10 | 3.15 | 3.14 | 0.915, 0.919, 0.937 |
| Echo 0.9 s, babble at 5 dB SNR | 1.07 | 2.78 | 2.70 | 0.657, 0.198, 0.328 |

Against Adobe's stems: isolated speech 21.3 dB SI-SDR, music 12.5 dB, background 4.6 dB.

## Known limits

- **Three voices:** the low balance, clarity curve, gate thresholds and highs guards rest on three reference voices plus synthetic noise.
- **Low harmonics:** `server/clarity.py` closes most of the gap to Adobe's contrast curve, but HNR and CPPS stay a little short on 1102 and Audio, and CPPS well short on the sample.
- **Heavy noise and clipping:** regenerated speech loses words the raw recording still has (the babble row). The gate backs off there, and Whisper reads the raw recording as well and keeps the one it is more confident of, which picked the better source on all nine recordings tested, one of them a near tie. The enhanced stem stays regenerated. How Adobe behaves in that range is unknown. These rules count a low speech share only when the fallback carried the voice: a voice 10 dB under real music reads as low as babble, yet BandIt keeps it and the enhanced stem transcribes at 0.93 against 0.56 for the raw mix, while BandIt files babble and distortion under "music" and loses the voice.
- **Music passages:** Sidon can turn a loud cymbal-like hit left in the speech stem into a short hiss.
- **Voice filed under music:** when BandIt does that, the music and background stems both hold the voice, in opposite polarity. They cancel at equal slider positions.
- **Inversion in part of a file:** a channel wired out of phase is caught per file, not per stretch.
- **v1 downloads:** stay at unity, because no Adobe v1 file was available.
- **Sparse files:** under about 15% speech, the highs step leaves Sidon's highs alone.
- **Phone-line stretches:** a stretch under 45% of a mostly wideband file gets its regenerated highs cut by up to 12 dB.
- **Quiet second talker:** a talker more than 30 dB under the main voice is past both gates' reach.
- **Memory after a job:** the models are dropped when the queue empties, but the system can hold about 2 GB of freed pages for a while.
- **Quiet second talker in a normal recording:** the lift is one gain for the whole file, so DeepFilterNet can still remove a talker under about -40 dBFS when the file's main voice is at a normal level.
- **A near talker 30 dB or more over the main voice for over 5% of the file:** the main voice then falls outside the voiced frames the level is read from, so the near talker counts as the speech. This comes from reading the code and was not tested.

## Open findings

The last independent review was stopped at the owner's request, so no second reviewer checked these. The reviewer measured most of them on saved stems or by direct calls, and a few by reading the code. Most need an unusual file or a failure; the download copies apply to any file. Fixed from that review: a key typed in the server's terminal cutting a file short, a failed job killing the worker, double-click delete taking the next track too, `ui/mock_shot.py` deleting real tracks, and the loud moments above.

Sound, most likely to be noticed first:

- **A hit no louder than the speech can still move the highs:** `highs.correction` is a power mean, and a broadband hit reads as a fricative. 0.1 s at 5 dB over the speech moved one voice's 10 kHz gain by 3.7 dB.
- **Interviews with two different sources:** a codec talker can take a wideband talker's highs boost, about +9 dB at 8-12 kHz (`highs.py`).
- **A long, loud music intro:** it can switch off the limit and low gate for the whole file, because `level.speech_share`'s whole-file figure is a power mean (`pipeline.py`).
- **Hiss on part of a recording:** it tightens the limit on the clean part too (`limit.py`).
- **Whispers:** the speech gate's breath rule can remove them. Tested on a synthetic whisper only.

Running the app:

- **First download fails:** a first v1 job can run on random weights if its model download fails. A truncated BandIt `model.pt` fails every v2 job until it is deleted.
- **Smaller ones:** Ctrl-C during an ffmpeg stage marks the job failed, WAV exports past 4 GB are broken (over about 4 h 8 min), every download leaves a copy in the track's folder until the track is deleted, `run.sh`'s page-opening loop can outlive a failed start, and deleting a track does not stop its transcript.
- **Bad samples:** one NaN or inf sample in a float WAV fails the job with a generic error.
- **`ui/e2e.py`, `ui/long_test.py`:** they crash on a failed job instead of reporting it.

## What was rejected

- **Fixed tone target:** fitted on one voice, it was the largest single error on the other.
- **Sidon's own waveform below 8 kHz, and grafting the recording's phase onto it:** moved pitch twice as far as Adobe does and matched its waveform less well than the recording's waveform with Sidon's envelope.
- **Feeding Sidon the denoised stem:** blurred weak syllables.
- **Fixed gap expander:** left the gaps well short of Adobe's near silence, most of all just before words.
- **Resemble Enhance:** further from Adobe than Sidon on both tests, and much slower.
- **Demucs:** BandIt separates speech, music and effects separately, which maps directly onto the sliders.
- **BandIt Plus:** quality matched BandIt v2 but its weights are non-commercial.
- **MossFormer2 after separation:** no measurable gain (3.35 vs 3.34), about a quarter more processing time. It stays in the quick v1 chain.
- **GPU (MPS):** slower than CPU for BandIt on an M1 Pro.
- **A correction learned from Adobe's outputs:** did not hold up on a voice it was not fitted on, and using those outputs as training data raises a terms question nobody has answered.
- **A new tone balance:** no better across the three voices.
- **Averaging several Sidon runs:** useless, because 96% of its error is systematic.

## Scripts

- `eval_pipeline.py <file> <tag> [v1|v2]` runs the real pipeline and reports timing. Stems stay in `work/<tag>/`.
- `scorecard.py <ours> <reference> [<isolated stem>]` produces the first table. Each file is a mono stem, a stereo download or a `merged.mp4`. Needs `praat-parselmouth` and `pyloudnorm`.
- `final_scores.py <tag>=<file>[|<clean reference>] ...` produces the results table. Needs `speechmos`.
- `score_stems.py <ours.mp4> <adobe.mp4>` compares two 8-channel files stem by stem. Needs `speechmos`.
- `seam_check.py <merged.mp4> <block seconds> ...` looks for clicks and level steps at block seams.
- `wysiwyg_check.py <merged.mp4>` confirms a download equals the player's gain formula with the mono speech stems 3 dB down in v2 (difference about -143 dBFS).
- `loud_moment_check.py <speech> <near talker> [dB]` adds a word close to the mic and compares the talker with the clean run.
- `sync_check.py <source> <merged.mp4>` confirms an exported video's isolated, music and background audio stays in sync (0 samples offset). Needs a video source.
- `make_sample.py <source> <start> <seconds>` rebuilds the landing page sample from the LibriVox reading.
- `ui/channel_test.py` checks channel routing in real Chrome.
- `ui/e2e.py`, `ui/edge.py` and `ui/long_test.py` drive the real app. They expect your own test recordings in `bench/real/`.
- `ui/mock_shot.py` recaptures the workspace picture on the landing page. It refuses to run unless the server's list is empty.

UI scripts use Playwright with the installed Google Chrome: `uv pip install playwright numpy`.

Speed on an M1 Pro, CPU only: a 36-second clip took 74 seconds with nothing else running, model loading included, and a 6.5-minute file took 14 minutes 39 seconds with other work running. So a 10-minute file takes about 20 minutes. Memory peaked at an 8.8 GB footprint. A job holds about 20 GB per hour of audio on disk at its peak, and about 3 GB per hour stays afterwards as `merged.mp4`.
