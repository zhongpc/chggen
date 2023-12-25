from torch import nn

def weight_init(model):
    '''Initialize weights and bias for the model.

    Args:
        model: model to be initialized.
    '''

    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0)

    elif isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
     
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)