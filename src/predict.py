import joblib

model = joblib.load("models/fraud_model.pkl")

def predict(data):
    prediction = model.predict([data])
    return int(prediction[0])