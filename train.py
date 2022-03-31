import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores




def train_model(train_dataloader, val_dataloader, model, n_epochs, save_loc):

    optimizer = torch.optim.Adam(list(model.parameters()), lr=5e-4)
    history = dict(train=[], val=[])
    best_loss = 10000.0

    for epoch in range(n_epochs):
        total_loss = 0.0
        print('Starting Epoch: {}'.format(epoch))
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for batch_quest, batch_ans, batch_vis in train_dataloader:
            batch_quest, batch_ans, batch_vis = batch_quest.to(device) , batch_ans.to(device) , batch_vis.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            predicted = model(batch_vis, batch_quest, batch_ans)
            loss = instance_bce_with_logits(predicted, batch_ans).to(device)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss: {0} - Epoch: {1}".format(round(epoch_loss,8), epoch+1))
        val_loss = test_model(val_dataloader, model)
        history['val'].append(val_loss)
        history['train'].append(epoch_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, save_loc+'/best_model.pth')
    print(history)
    return model, history


