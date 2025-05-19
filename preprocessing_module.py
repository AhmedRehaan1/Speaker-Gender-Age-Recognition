import os
import pandas as pd
import librosa
import numpy as np

class PreprocessingModule:
    def load_data_set(dataset_path):
        '''
        loads all .mp3 and .wav file under the directory with path dataset_path
        '''
        data = pd.DataFrame(columns=['full_path'])
    
        audio_root = os.path.join(dataset_path)
        filename_to_path = {}
    
        for root, dirs, files in os.walk(audio_root):
            for f in files:
                if f.endswith(".mp3") or f.endswith(".wav"):
                    file_path = os.path.join(root, f)
                    if os.path.getsize(file_path) > 0:  # Ignore zero-byte files
                        filename_to_path[f] = file_path
                        # Append to DataFrame
                        data.loc[len(data)] = [file_path]
                    else:
                        print(f"Skipped zero-byte file: {file_path}")
    
        data['file_exists'] = data['full_path'].apply(lambda p: p is not None and os.path.exists(p))
        print("Found files:", data['file_exists'].sum())
        print("Missing files:", (~data['file_exists']).sum())
    
        data = data.dropna()
        return data
    
    def preprocess_audio(file_path, target_sr=16000, duration=3):
        '''
        target_sr: sample rate for all audio files
        duration: desired length of final audio in seconds
        '''
        # load audio file with its original SR then sample at tareget_sr
        y, sr = librosa.load(file_path, sr=None, mono=True)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
        # Normalizes to values between -1 and 1
        y = y / np.max(np.abs(y))
    
        # Trims silence (less than 20 dB)
        y, _ = librosa.effects.trim(y, top_db=18)
    
        # Pads with zeros if shorter than duration, or truncates if longer (??)
        y = np.pad(y, (0, max(0, duration * target_sr - len(y))))[:duration * target_sr]
    
        return y, target_sr


