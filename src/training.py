def train_model(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        loss_sum = 0
        batch_sum = 0
        
        for images, labels in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_sum += 1
        
        print(f"Epoch: {epoch + 1} - Average loss: {loss_sum / batch_sum}")
        
    return model