
from matplotlib import pyplot as plt
from DLSnow.evaluation import accuracy
import DLSnow.functions as Fun
import DLSnow.datasets
from DLSnow import optimizers
from net.ANN import ANN

# super param
max_epoch = 1000
batch_size = 30
hidden_size = 50
learning_rate = 0.1

# data load
train_set = DLSnow.datasets.Spiral(train=True)
test_set = DLSnow.datasets.Spiral(train=False)
train_loader = DLSnow.DataLoader(train_set, batch_size)
test_loader = DLSnow.DataLoader(test_set, batch_size, shuffle=False)

model = ANN((hidden_size, hidden_size, 3), activation=Fun.relu)
optimizer = optimizers.Adam(learning_rate)
optimizer.setup(model)

test_loss_history = []
test_acc_history = []
train_loss_history = []
train_acc_history = []


# train
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = Fun.softmax_cross_entropy(y, t)
        acc = accuracy(y, t)
        model.clear_grad()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_acc_history.append(sum_acc/len(train_set))
    train_loss_history.append(sum_loss/len(train_set))
    print('Epoch: {}'.format(epoch+1))
    print("train : Loss: {}, \tAccuracy: {}".format(sum_loss/len(train_set),sum_acc/len(train_set)))

    sum_loss, sum_acc = 0, 0
    with DLSnow.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = Fun.softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss_history.append(sum_loss / len(test_set))
    test_acc_history.append(sum_acc / len(test_set))
    print("test : Loss: {}, \tAccuracy: {}".format(sum_loss/len(train_set),sum_acc/len(train_set)))


# plot the loss and acc
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.plot(train_loss_history, label='Training Loss')
plt.plot(test_loss_history, label='Testing Loss')
plt.show()

plt.xlabel('Iteration')
plt.ylabel('ACC')
plt.title('Training Acc Curve')
plt.plot(train_acc_history, label='Training Acc')
plt.plot(test_acc_history, label='Test Acc')
plt.show()
