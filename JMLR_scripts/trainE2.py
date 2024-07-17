"""

Script used for the experiments as reported in the JMLR paper

Author: Valentin Delchevalerie (UNamur)

"""


import sys
from utils import getData, WarmupCosineSchedule
from generateE2 import loadModel
import torch
import torch.nn.functional as F
import time
import os
import json

import gc
gc.collect()

#import torchvision
#import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Get the arguments """
model_name, dataset_name, train_size, run_id = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
print(model_name, dataset_name, train_size, run_id)

""" Generate the saving folder """
save_folder = './results/' + dataset_name + '/' + model_name + '/' + str(train_size) + '/' + str(run_id) + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

""" Get the dataloader """
train_dataset, test_dataset = getData(dataset_name, train_size, run_id)

""" Load the model """
model, training_params = loadModel(dataset_name, model_name)
model.to(device)
pytorch_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('\n\nThe model is made of', pytorch_n_params, 'trainable parameters\n\n')

pytorch_n_params = 0
for layer in model.children():
    for sub_layer in layer.children():
        if 'Bessel' in str(sub_layer):
            pytorch_n_params += sub_layer.n_params
        else:
            pytorch_n_params += sum(p.numel() for p in sub_layer.parameters() if p.requires_grad)

print('\n\nThe model is made of', pytorch_n_params, 'trainable parameters\n\n')  

""" Training loop """

if train_size < 0.01:
    n_epochs = training_params['epochs'] * 3
else:
    n_epochs = training_params['epochs']

warmups_steps = training_params['epochs'] // 5
scheduler = WarmupCosineSchedule(optimizer=training_params['optimizer'], warmup_steps=warmups_steps, t_total=n_epochs)

stats = {}
for epoch in range(n_epochs):

    stats[str(epoch)] = {}
    print("Epoch {}/{}".format(epoch + 1, n_epochs))

    # Training 
    model.train()

    train_loss = 0.
    train_metrics = [0.] * len(training_params['metrics'])

    tot_time = 0
    training_time = 0
    testing_time = 0

    tic_tot_time = time.time()
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        images, labels = x_batch_train.to(device), y_batch_train.to(device)
        labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

        #if step == 0:
        #    grid = torchvision.utils.make_grid(images[:10], nrow=2)
        #    torchvision.utils.save_image(grid.cpu(), save_folder+'training_'+str(epoch)+'.png')

        tic_training_time = time.time()

        training_params['optimizer'].zero_grad()
        logits = model(images)

        #loss = training_params['loss'](F.log_softmax(logits, dim=1), labels)
        loss = training_params['loss'](logits, labels)

        loss.backward()
        training_params['optimizer'].step()

        training_time += time.time() - tic_training_time

        for i, metric in enumerate(training_params['metrics']):
            train_metrics[i] += metric(logits, labels).item() / len(train_dataset)
        train_loss += loss.item() / len(train_dataset)
    
    scheduler.step()

    # Validation
    model.eval()

    val_loss = 0.
    test_metrics = [0.] * len(training_params['metrics'])
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

        images, labels = x_batch_test.to(device), y_batch_test.to(device)
        labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

        #if step == 0:
        #    grid = torchvision.utils.make_grid(images[:10], nrow=2)
        #    torchvision.utils.save_image(grid.cpu(), save_folder+'testing_'+str(epoch)+'.png')

        tic_testing_time = time.time()

        logits = model(images)

        #loss = training_params['loss'](F.log_softmax(logits, dim=1), labels)
        loss = training_params['loss'](logits, labels)

        testing_time += time.time() - tic_testing_time

        for i, metric in enumerate(training_params['metrics']):
            test_metrics[i] += metric(logits, labels).item() / len(test_dataset)
        val_loss += loss.item() / len(test_dataset)

    tot_time = time.time() - tic_tot_time

    print("Training metrics:", train_metrics, "; Training loss: %.4f" % float(train_loss), 
          "; Testing metrics:", test_metrics, "; Testing loss: %.4f" % float(val_loss), 
          "; Learning rate: %.6f" % float(training_params['optimizer'].param_groups[0]['lr']))
    stats[str(epoch)] = {'Training metrics': [float(x) for x in train_metrics], 'Training loss': float(train_loss),
                         'Testing metrics': [float(x) for x in test_metrics], 'Testing loss': float(val_loss),
                         'lr': float(training_params['optimizer'].param_groups[0]['lr']),
                         'tot_time': float(tot_time),
                         'training_time': float(training_time),
                         'testing_time': float(testing_time)}
    print("Time taken: %.2fs" % (tot_time))

with open(save_folder + 'stats.json', 'w') as f:
    json.dump(stats, f, indent=8)