from huggingface_hub import snapshot_download
import sys

MODEL_ID = "shalev396/california-housing"


def load_model():
    path = snapshot_download(MODEL_ID)
    sys.path.insert(0, path)

    import model

    return model.load(path, device="cpu")



def predict(model, housing_data):
    prediction = model.predict(housing_data)
    return prediction