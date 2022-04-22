import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_test_custom_image as tci
from torchvision import datasets, transforms

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Image is 28x28 so 748 total pixels
        self.input_layer = nn.Linear(784, 64)
        self.hidden_layer1 = nn.Linear(64, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden_layer1(data))
        data = F.relu(self.hidden_layer2(data))
        data = self.output_layer(data)

        return F.log_softmax(data, dim=1)


if __name__ == '__main__':
    # download the testing and training datasets
    training = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    testing = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Import the testing and datasets into code
    train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)
    test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)

    # Initialize the network and set parameters
    network = Network()
    learn_rate = optim.Adam(network.parameters(), lr=0.001) # Mess with lr to get best results
    epochs = 10  # Number of training cycles for the network

    # Train the network
    print('Training network...')
    for i in range(epochs):
        for data in train_set:
            # Get the image and the expected output
            image, output = data

            # Reset the network's gradient (makes each image unique)
            network.zero_grad()

            # Run the image through the network
            result = network(image.view(-1, 784)) # Export all data in a 784 entry array

            # How far off the network's guess is
            loss = F.nll_loss(result, output)

            # Update the network's weights through backward propagation
            loss.backward()
            learn_rate.step()
        print(loss)

    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        print('Running downloaded test set...')
        for data in test_set:
            # Get the image and the expected output
            image, output = data

            # Run the image through the network
            result = network(image.view(-1, 784)) # Export all data in a 784 entry array

            for index, tensor_value in enumerate(result):
                # For each result in the batch, check whether the guess was correct
                total += 1

                if torch.argmax(tensor_value) == output[index]:
                    correct += 1

    accuracy = (correct / total) * 100.0
    print(f'Accuracy: {accuracy}%\n')

    tci.test_custom_image('Test.png', network)