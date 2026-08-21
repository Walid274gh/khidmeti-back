
STT Data Expansion Strategy (target 15k+ clips):

1. YouTube scraping (est. 8k clips):
   - Algerian home repair channels: "سباكة دار", "كهرباء منزلية"
   - Moroccan service vlogs: "خدمات منزلية", "صيانة"
   - Use yt-dlp + VAD segmentation + Whisper for validation
   - Filter: 0.5-30s clips, clear speech, service-related keywords

2. Algerian radio/podcast archives (est. 3k clips):
   - Radio Algérienne call-in shows
   - Chaine 3 archives (mixed Arabic/French like Darija)
   - Segment with VAD, transcribe with current model for filtering

3. Synthetic TTS augmentation (est. 2k clips):
   - Festival/eSpeak with Arabic phonemes
   - Read NLU training texts for aligned data
   - Only use for augmentation, never validation

4. Crowd-sourced recording (est. 1k clips):
   - In-app: record service request after submission
   - Consent-gated (already wired in C1 flywheel)
   - Monthly batches for P6 retraining

5. Data augmentation (est. 1k additional):
   - Speed: 0.9x, 1.1x variants
   - Pitch: ±2 semitones
   - Noise: cafe/street ambient at SNR 15-25dB
   - librosa augmentations on base clips

Implementation notes:
- All sources require manual spot-checking (10% sample)
- Maintain split: new data goes to train only, heldout stays frozen
- Track provenance: source field per clip for ablations
- License check: YouTube CC-BY only, radio requires permission
- Quality gate: WER on known-good subset < 0.60 before serving

Tools needed:
- yt-dlp for YouTube download
- webrtcvad for speech segmentation
- librosa/audiomentations for augmentation
- Manual transcription tool for 1k validation clips

Timeline: 2-3 weeks of data collection + 1 week processing + 2 days retraining
