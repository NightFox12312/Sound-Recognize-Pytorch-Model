import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Define the model
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[-1])
        return output

# Define the dataset and dataloader
class SpeechRecognitionDataset(data.Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio = self.audio_files[index]
        label = self.labels[index]
        return audio, label

train_dataset = SpeechRecognitionDataset(audio_files, labels)
train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the loss function and optimizer
model = SpeechRecognitionModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(num_epochs):
    for audio, label in train_dataloader:
        optimizer.zero_grad()
        output = model(audio)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# Test the model
test_dataset = SpeechRecognitionDataset(test_audio_files, test_labels)
test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0
    for audio, label in test_dataloader:
        output = model(audio)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print("Accuracy:", 100 * correct / total)

# Save the model
torch.save(model.state_dict(), "speech_recognition_model.pth")
