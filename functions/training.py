import torch.nn.functional as F
import torch


# train the model
def train_model(train_loader, device, model, optimizer, epoch,  EPOCHS, BATCH_SIZE):
    model.train()
    losses = []
    # print("<=========== EPOCH "+ str(epoch+1) +" ===========>")
    # enumate batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data, target = data.to(device), target.to(device)
        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.cross_entropy(y_pred, target)
        losses.append(loss.cpu().item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        # update model weights
        optimizer.step()

        # Display
        if batch_idx % 30 == 1:
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1,
                EPOCHS,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / BATCH_SIZE,
                loss.cpu().item()),
                end='')

    return losses


def evaluate_model(model, device, test_loader, loss, EPOCHS, epoch, batch_size, train_size):
    # display final evaluation for this epoch
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)

        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
            epoch + 1,
            EPOCHS,
            train_size,
            train_size,
            100* batch_size / batch_size,
            loss,
            accuracy * 100,
            end=''))

    return accuracy