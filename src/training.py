import torch

def train_model(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        loss_sum = 0
        batch_sum = 0
        samples = 0
        true_predictions = 0
        
        for images, labels in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_sum += 1
            samples += len(labels)
            predictions = torch.argmax(output, dim=1)
            true_predictions += (predictions == labels).sum().item()
        
        print(
            f"Epoch: {epoch + 1} - "
            f"Loss: {loss_sum / batch_sum} - "
            f"Accuracy: {true_predictions / samples}"
        )
        
    return model

def validate_model(model, dataloader, loss_function):
    model.eval()
    loss_sum = 0
    samples = 0
    true_predictions = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            output = model(images)
            loss = loss_function(output, labels)
            loss_sum += loss.item()
            predictions = torch.argmax(output, dim=1)
            samples += len(labels)
            true_predictions += (predictions == labels).sum().item()
            print(f"Labels: {labels}")
            print(f"Predictions: {predictions}")

    print(f"Average loss: {loss_sum / len(dataloader)}")
    print(f"Samples: {samples}")
    print(f"True predictions: {true_predictions}")
    print(f"Accuracy: {true_predictions / samples}")