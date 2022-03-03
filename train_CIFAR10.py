from project1_model import ResNet, BasicBlock 
import numpy as np, torch, torchvision, os 
import matplotlib.pyplot as plt 

trainingdata = torchvision.datasets.CIFAR10('./CIFAR10/', 
                                                 train=True, 
                                                 download=True, 
                                                 transform=torchvision.transforms.ToTensor()) 

testdata = torchvision.datasets.CIFAR10('./CIFAR10/', 
                                             train=False, 
                                             download=True, 
                                             transform=torchvision.transforms.ToTensor()) 

trainDataLoader = torch.utils.data.DataLoader(trainingdata, 
                                              batch_size=64, 
                                              shuffle=True)

testDataLoader = torch.utils.data.DataLoader(testdata, 
                                             batch_size=64, 
                                             shuffle=False) 
images, labels = iter(trainDataLoader).next() 

net = ResNet(BasicBlock, [2, 2, 2, 2]) 
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

EPOCHS = 100 

train_loss_history = []
test_loss_history = []
accuracy_history = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    test_loss = 0.0
    for i, data in enumerate(trainDataLoader):
        images, labels = data
        images = images 
        labels = labels 
        optimizer.zero_grad()
        predicted_output = net(images)
        fit = Loss(predicted_output,labels)
        fit.backward()
        optimizer.step()
        train_loss += fit.item()
    for i, data in enumerate(testDataLoader):
        with torch.no_grad():
            images, labels = data
            images = images 
            labels = labels 
            predicted_output = net(images)
            fit = Loss(predicted_output,labels)
            test_loss += fit.item()
    train_loss = train_loss/len(trainDataLoader)
    test_loss = test_loss/len(testDataLoader)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    _, indices = torch.max(predicted_output, 1)
    accuracy = torch.tensor(torch.sum(indices == labels).item() / len(predicted_output))
    accuracy_history.append(accuracy) 

    if epoch % 10 == 0: 
        ckpt_dir = 'models/run1/checkpoints/' 
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir) 
        torch.save(net.state_dict(), ckpt_dir+'full_resnet_'+str(epoch)+'.pt') 

    print('Epoch %s, Train loss %.3f, Test loss %.3f, Accuracy %.3f'%(epoch, train_loss, test_loss, accuracy))

plt.plot(range(EPOCHS),train_loss_history,'-',linewidth=3,label='Train error')
plt.plot(range(EPOCHS),test_loss_history,'-',linewidth=3,label='Test error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.savefig('models/run1/loss_plot')
plt.close()

plt.plot(range(EPOCHS),accuracy_history,'-',linewidth=3,label='Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend()
plt.savefig('models/run1/accuracy_plot')
plt.close()



