import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import string
from gtts import gTTS  # <--- New import for TTS
import os
import streamlit as st
import webbrowser

# ===== Setup =====
CHARS = string.ascii_lowercase + string.digits  # a-z + 0-9
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
NUM_CLASSES = len(CHARS) + 1  # CTC blank

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Model Definition =====
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(256 * 2, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)

# ===== Decoder =====
def decode(logits):
    probs = torch.softmax(logits, dim=2)
    _, preds = probs.max(2)
    preds = preds.permute(1, 0)  # [B, T]
    results = []
    for pred in preds:
        string = ''
        last = -1
        for p in pred:
            p = p.item()
            if p != last and p != len(CHARS):
                string += idx_to_char[p]
            last = p
        results.append(string)
    return results

# ===== Image Preprocessor =====
def process_image(image):
    image = ImageOps.grayscale(image)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.resize((100, 32))
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(DEVICE)

# ===== Text-to-Speech =====
def text_to_speech(text, filename="prediction_audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# ===== Streamlit App =====
st.title("📝 Optical Character Recognition System (CRNN + CTC)")
st.write("Upload an image of a word (printed) and let the model try to read it.")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load model
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load("crnn_model.pth", map_location=DEVICE))
    model.eval()

    # Predict
    input_tensor = process_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = decode(output)[0]

    # Show prediction
    st.subheader("📖 Predicted Text:")
    if prediction.strip() == "":
       # st.write("Raw logits shape:", output.shape)
       # st.write("Decoded logits:", decode(output))
        st.warning("No text could be confidently predicted from the image.")
        probs = torch.softmax(output, dim=2)
        _, preds = probs.max(2)
        st.write("Predicted class indices:", preds.squeeze().tolist())
    else:
        st.code(prediction)
        #st.write("Raw logits shape:", output.shape)
        #st.write("Decoded logits:", decode(output))

        # 🔊 Read Aloud Button
        if st.button("🔊 Read Prediction Aloud"):
            audio_path = text_to_speech(prediction)
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")



# ====================== DIGIT RECOGNITION SECTION ======================


