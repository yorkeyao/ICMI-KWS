import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss()

    def forward(self, input, target):
        loss = 0.0
        transposed = input.transpose(0, 1).contiguous() # log-probabilities 

        for i in range(0, 29):
            loss += self.criterion(transposed[i], target)

        return loss

def _validate_pred(modelOutput, length, total=None, wrong=None):
    
    # print (np.shape (modelOutput))
    averageEnergies = torch.sum(modelOutput.data, 1)
    # print (np.shape (averageEnergies))

    # assert (0)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        # print (i)
        # print (length[i])
        # print ( np.shape (modelOutput))
        # print ( np.shape ( modelOutput[i,:length[i]].sum(0)) )
        averageEnergies[i] = modelOutput[i,:length[i]].sum(0)
        # assert (0)
    # print (np.shape (averageEnergies))
    # assert (0)
    maxvalues, maxindices = torch.max(averageEnergies, 1)

    return (maxvalues, maxindices, averageEnergies)
    

def _validate(modelOutput, length, labels, total=None, wrong=None):
    
    
    averageEnergies = torch.sum(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        averageEnergies[i] = modelOutput[i,:length[i]].sum(0) / length[i]
    #print (np.shape (averageEnergies))
    #assert (0)
    maxvalues, maxindices = torch.max(averageEnergies, 1)

    # print (np.shape (averageEnergies))
    # print (averageEnergies[0])

    # print (averageEnergies[1])
    # print (maxvalues[0:5])

    # assert (0)
    count = 0

    

    error_list = []

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1 
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
            error_list.append(1)
        else:
            error_list.append(0)
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1
    

    # print (error_list[0: 5])
    
    error_list = np.array(error_list)

    # print (np.where(error_list == 0))

    # print (maxvalues[np.where(error_list == 0)])

    print ( "true", np.average( maxvalues[np.where(error_list == 1)].cpu().numpy() ))

    print ( "false", np.average( maxvalues[np.where(error_list == 0)].cpu().numpy() ))

    return (averageEnergies, count)

class LSTMBackend(nn.Module):
    def __init__(self, options):
        super(LSTMBackend, self).__init__()
        self.Module1 = nn.LSTM(input_size=options["model"]["inputdim"],
                                hidden_size=options["model"]["hiddendim"],
                                num_layers=options["model"]["numlstms"],
                                batch_first=True,
                                bidirectional=True)

        self.fc = nn.Linear(options["model"]["hiddendim"] * 2,
                                options["model"]["numclasses"])

        self.softmax = nn.LogSoftmax(dim=2)

        self.loss = NLLSequenceLoss()

        self.validator = _validate

        self.validator_pred = _validate_pred

    def forward(self, input):

        temporalDim = 1

        lstmOutput, _ = self.Module1(input)

        output = self.fc(lstmOutput)
        output = self.softmax(output)

        return output
