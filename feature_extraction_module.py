import librosa
import pandas as pd
import numpy as np

class FeatureExtractionModule:
    def extract_features(signal , sample_rate, mfcc_coeff_num=20):
        '''
            returns the following features:
                mfcc_coeff_num MFCC coefficients mean(s) and standard deviation(s) across time for the given signal
                one pitch (fundamental frequency) mean and one standard deviation across time for the given signal
                one spectral_centroid mean across time for the given signal
                seven spectral_contrast mean(s) across time for the given signal (for the default seven frequency bands)
                twelve chroma mean(s) across time for the given signal
                one zero crossing rate mean across time for the given signal
        '''
        # MFCC coefficients : male and female voices have distinct MFCC patterns. Also , age related changes may affect MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=mfcc_coeff_num) # has shape (n_mfcc, num_frames)
        mfccs_mean = np.mean(mfccs, axis=1) # calculating along axis 1 in order to aggeragte feature values across time frames
        mfccs_std = np.std(mfccs, axis=1)
    
        # Pitch (or fundamental frequency f0) : ranges of pitch are different for males and females
        pitch = librosa.pyin(signal, fmin=50, fmax=500, sr=sample_rate)[0] # voicing probability and voicing probability are not needed here
        pitch_mean = np.nanmean(pitch) if any(pitch) else 0 # to ignore NaN values
        pitch_std = np.nanstd(pitch) if any(pitch) else 0
    
        # Spectral centroid : Measures the "center of mass" of the spectrum, indicating brightness or clarity of the sound.
        # Younger voices may sound brighter, while older voices may have a "duller" quality due to vocal changes.
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
        spectral_centroid_mean = np.mean(spectral_centroid)
    
        # Spectral Contrast : Captures the difference in amplitude between peaks and valleys in the spectrum, reflecting voice
        # clarity and resonance. This can differ by gender (due to formant differences) and age (due to changes in vocal resonance).
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1) # mean is calculated along axis 1
    
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
    
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(signal)
        zcr_mean = np.mean(zcr)
    
        # Combine all features
        features = np.concatenate([
        mfccs_mean, mfccs_std,  # 2 * n_mfcc features
        [pitch_mean, pitch_std],  # 2 features
        [spectral_centroid_mean],  # 1 feature
        spectral_contrast_mean,  # 7 features (default bands)
        chroma_mean,  # 12 features
        [zcr_mean]  # 1 feature
        ])
        return features

    def get_feature_names(mfcc_coeff_num=20):
        feature_names = (
            [f'mfcc_{i}_mean' for i in range(mfcc_coeff_num)] +  # 20 MFCC means
            [f'mfcc_{i}_std' for i in range(mfcc_coeff_num)] +   # 20 MFCC stds
            ['pitch_mean', 'pitch_std'] +                       # 2 pitch features
            ['spectral_centroid_mean'] +                        # 1 centroid feature
            [f'contrast_{i}' for i in range(7)] +               # 7 contrast features
            [f'chroma_{i}' for i in range(12)] +                # 12 chroma features
            ['zcr_mean']                                        # 1 ZCR feature
        )
        return feature_names
