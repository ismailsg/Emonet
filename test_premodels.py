import torch
from torchvision import transforms
from PIL import Image
from emonet.models import EmoNet
import numpy as np

#https://github.com/tgberkeley/EmoFAN-VR/tree/main

pretrained_model_path = '/home/ismail/Documents/Emonet_vad/emonet_8.pth'
pretrained_model = torch.load(pretrained_model_path)

# Rename the keys to match the expected keys of the current EmoNet model
renamed_pretrained_model = {
    k.replace('emo_fc_2.3.', 'emo_fc_2.4.'): v for k, v in pretrained_model.items()
}

# Load the modified pretrained model
net = EmoNet(n_expression=8)
net.load_state_dict(renamed_pretrained_model)
net.eval()

# Preprocess the image
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = '/home/ismail/Documents/images/12.jpg'
image = Image.open(image_path)
image = transform_image(image).unsqueeze(0)  # Add batch dimension

# Perform prediction
with torch.no_grad():
    prediction = net(image)

    

valence = prediction['valence']
arousal = prediction['arousal']
expression =prediction['expression']



val = np.squeeze(valence.cpu().numpy())
ar = np.squeeze(arousal.cpu().numpy())
    

print("Predicted Valence:", val)
print("Predicted Arousal:", ar)



