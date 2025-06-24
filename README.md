# DESCRIPTION

This project is a complete, offline speaker diarisation and transcription pipeline built in Python. It leverages state-of-the-art models, using pyannote.audio for speaker diarisation and OpenAI's Whisper for transcription, to produce highly accurate transcripts that identify "who said what".

The core architecture is designed to be memory-efficient, sequentially loading and unloading AI models to enable large models (like whisper-large-v3) to run on consumer-grade GPUs with limited VRAM (e.g., 8GB).

# KEY FEATURES

High-Accuracy Transcription: Utilises OpenAI's powerful Whisper models (configurable size from tiny to large-v3) for state-of-the-art speech-to-text conversion.
Speaker Diarisation: Pinpoints who spoke and when using pyannote.audio, with the ability to provide hints for the number of speakers.
Offline & Private: The entire pipeline runs locally. Your audio files are never uploaded to the cloud.
VRAM-Efficient Architecture: Employs a sequential processing model that loads and unloads AI components one at a time, making it possible to run large models on consumer GPUs with as little as 8GB of VRAM.
Modular & Extensible: Built with a clear separation of concerns, with dedicated processors for each stage (Preprocessing, VAD, Diarisation, Transcription), making the codebase easy to maintain and extend.
Audio Preprocessing: Includes an initial step for noise reduction and volume normalisation to improve the accuracy of the AI models.
Flexible Input: Capable of processing common audio and video formats (e.g., MP4, M4A, WAV, MP3) via FFmpeg.