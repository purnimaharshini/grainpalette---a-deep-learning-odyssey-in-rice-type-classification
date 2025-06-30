from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)