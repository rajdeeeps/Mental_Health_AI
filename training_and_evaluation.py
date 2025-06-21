# Loading Dependencies
from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Initialize BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Evaluation function
def eval_model(model, data_loader):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    return acc, f1

# Run Training Loop
epochs = 3
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}")

    train_loss = train_epoch(model, train_loader, optimizer)
    acc, f1 = eval_model(model, test_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")