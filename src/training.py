import torch


def train_model(model, training_dataloader, validation_dataloader, loss_function, optimizer, epochs):
    
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        batch_sum = 0
        samples = 0
        true_predictions = 0
        
        for images, labels in training_dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            batch_sum += 1
            samples += len(labels)
            predictions = torch.argmax(output, dim=1)
            true_predictions += (predictions == labels).sum().item()

        validation_loss, validation_accuracy = validate_model(
            model=model,
            dataloader=validation_dataloader,
            loss_function=loss_function
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {loss_sum / samples:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Train acc: {true_predictions / samples:.2%} | "
            f"Val acc: {validation_accuracy:.2%}"
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
            loss_sum += loss.item() * len(labels)
            predictions = torch.argmax(output, dim=1)
            samples += len(labels)
            true_predictions += (predictions == labels).sum().item()

    average_loss = loss_sum / samples
    accuracy = true_predictions / samples
    
    return average_loss, accuracy