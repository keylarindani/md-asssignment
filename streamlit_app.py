import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class ObesityPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.load_data()
        self.preprocess_data()
        self.train_model()
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.X = self.data.drop('NObeyesdad', axis=1)
        self.y = self.data['NObeyesdad']
        
    def preprocess_data(self):
        # Identify categorical and numerical columns
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Preprocess features
        self.X_processed = self.preprocessor.fit_transform(self.X)
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
    def train_model(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y_encoded, test_size=0.2, random_state=42)
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
    def predict(self, input_data):
        # Preprocess input data
        input_processed = self.preprocessor.transform(input_data)
        
        # Make prediction
        probabilities = self.model.predict_proba(input_processed)[0]
        prediction = self.model.predict(input_processed)[0]
        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return probabilities, prediction_label
