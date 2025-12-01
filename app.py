import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

from your_model_file import BloodCellViT, PerformerModel, CustomViT, Generator  # Import all models

# -----------------------
# Initialize Flask App
# -----------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Models
# -----------------------
# ViT (pretrained)
vit_model = BloodCellViT(num_classes=4)
vit_model.load_state_dict(torch.load("models/vit_model.pth", map_location=device))
vit_model.to(device).eval()

# Custom ViT
custom_vit = CustomViT(num_classes=4)
custom_vit.load_state_dict(torch.load("models/custom_vit_model.pth", map_location=device))
custom_vit.to(device).eval()

# Performer
performer_model = PerformerModel()
performer_model.load_state_dict(torch.load("models/performer_model.pth", map_location=device))
performer_model.to(device).eval()

# GAN Generator
gan_generator = Generator()
gan_generator.load_state_dict(torch.load("models/gan_generator.pth", map_location=device))
gan_generator.to(device).eval()

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict with all models (you can choose which to use)
    with torch.no_grad():
        vit_pred = torch.argmax(vit_model(input_tensor), dim=1).item()
        custom_pred = torch.argmax(custom_vit(input_tensor), dim=1).item()
        performer_pred = torch.argmax(performer_model(input_tensor), dim=1).item()

    # Map class index to label
    classes = ["NEUTROPHIL", "EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE"]
    result = {
        "ViT": classes[vit_pred],
        "CustomViT": classes[custom_pred],
        "Performer": classes[performer_pred]
    }

    return jsonify(result)

@app.route("/generate_gan", methods=["GET"])
def generate_gan():
    z = torch.randn(1, 100, 1, 1).to(device)
    with torch.no_grad():
        fake_img = gan_generator(z)

    # Convert to PIL image
    fake_img = (fake_img.squeeze(0).cpu() + 1) / 2  # [-1,1] -> [0,1]
    to_pil = transforms.ToPILImage()
    img = to_pil(fake_img)
    save_path = "static/generated.png"
    img.save(save_path)

    return jsonify({"gan_image": save_path})

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
