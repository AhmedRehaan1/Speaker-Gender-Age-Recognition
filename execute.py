import pickle
import sys
from preprocessing_module import PreprocessingModule
from feature_extraction_module import FeatureExtractionModule
import pandas as pd
import joblib

def input_apply_function(path):
    signal , sr = PreprocessingModule.preprocess_audio(path)
    return FeatureExtractionModule.extract_features(signal, sr)


if __name__ == '__main__':
    gender_model_path = ''
    age_model_path = ''
    input_data_path = ''
    if len(sys.argv) < 4:
        print('model pickle file paths not provided . Assuming gender_model.pkl and age_model.pkl exists in the current directory')
        gender_model_path = 'gender_model.pkl'
        age_model_path = 'age_model.pkl'
        input_data_path = '.'
    else:
        gender_model_path = sys.argv[1]
        age_model_path = sys.argv[2]
        input_data_path = sys.argv[3]
    gender_model = pickle.load(open(gender_model_path, 'rb'))
    age_model = pickle.load(open(age_model_path, 'rb'))
    print(age_model)
    print(gender_model)
    data = PreprocessingModule.load_data_set(input_data_path)
    data = data.drop(columns=['file_exists'])
    data = data.dropna()

    features = data['full_path'].apply(input_apply_function)

    # Convert features to DataFrame
    feature_names = FeatureExtractionModule.get_feature_names(mfcc_coeff_num=20)
    features_df = pd.DataFrame(features.tolist(), columns=feature_names, index=data.index)
    

    with open('predictions.txt', 'w') as f:
        for idx, row in features_df.iterrows():
            age_prediction = age_model.predict([row])[0]
            gender_prediction = gender_model.predict([row])[0]
            f.write(f"{gender_prediction}   {age_prediction}\n")

    

