from sklearn.model_selection import train_test_split
from preprocessing_module import PreprocessingModule
from feature_extraction_module import FeatureExtractionModule
from model_training_module import ModelTrainingModule
import pandas as pd
import sys
import pickle

def apply_function(path):
    signal , sr = PreprocessingModule.preprocess_audio(path)
    return FeatureExtractionModule.extract_features(signal, sr)

if __name__ == '__main__':
    dataset_path = ''
    if len(sys.argv) <= 1:
        print('Missing path to dataset directory ("data" directory is assumed to exist in the current directory)')
        dataset_path = 'data'
    else:
        dataset_path = sys.argv[1]

    data = PreprocessingModule.load_data_set(dataset_path)
    features = data['full_path'].apply(apply_function)

    # Convert features to DataFrame
    feature_names = FeatureExtractionModule.get_feature_names(mfcc_coeff_num=20)
    features_df = pd.DataFrame(features.tolist(), columns=feature_names, index=data.index)
    
    # Concatenate with original DataFrame
    data = pd.concat([data, features_df], axis=1)
    data = data.drop(columns=['full_path'])
    data = data.dropna()

    gender_data = data.iloc[:100000, :]
    y_gender = gender_data['gender'] 
    x_gender = gender_data.drop(columns=['age', 'gender', 'label'])
    x_gender_resampled, y_gender_resampled = ModelTrainingModule.undersample(x_gender, y_gender)

    x_gender_train, x_gender_test, y_gender_train, y_gender_test = train_test_split(x_gender_resampled, y_gender_resampled, test_size=0.2, random_state=42)
    gender_model, params, score = ModelTrainingModule.tune_svm_hyperparameters(x=x_gender_train, y=y_gender_train, C_range=[0.1, 1, 10, 100,200])

    pickle.dump(gender_model, 'gender_model.pkl')


    y_age = data['age']
    x_age = data.drop(columns=['age', 'gender', 'label'])

    age_model = ModelTrainingModule.train_random_forest(x_age, y_age, n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42, test_split_random_state=6)
    pickle.dump(gender_model, 'age_model.pkl')