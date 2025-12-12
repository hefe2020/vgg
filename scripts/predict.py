import sys
from src.vgg.infer import load_model, predict

model, class_idx = load_model("artifacts/best_model.pt", "artifacts/class_index.json")

img_path = sys.argv[1]
label, score = predict(model, img_path, class_idx)

print(f"Predicted: {label} ({score:.3f})")

