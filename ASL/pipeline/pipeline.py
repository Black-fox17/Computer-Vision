import torch
import os
from models import ASLClassifier
import torch.nn as nn
from .utils import draw_landmarks,num_to_char
class ModelPipeline:
    def __init__(self,model_path):

        self.model = ASLClassifier()  # Create an instance of the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    def train(self,data_loader):
        for epoch in range(1):  # Adjust as needed
            self.model.train()
            total_loss = 0
            print(f"\n=== Epoch {epoch+1} ===")

            for batch_idx, (images, labels) in enumerate(data_loader):
                labels = labels.squeeze()  # If shape is [B, 1], make it [B]

                print(f"\nBatch {batch_idx + 1}")

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                print(f"Batch loss: {loss.item():.4f}")

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

    def inference(self,image):
        self.model.eval()
        frame = draw_landmarks(image)
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        pred = self.model(frame_tensor)
        result = torch.argmax(pred, dim=1)
        return num_to_char(result.tolist()[0])


model_pipeline = ModelPipeline(r"ASL\models\checkpoint.pth")
