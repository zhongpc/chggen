import torch

############################################################################
# Class for early stopping.
############################################################################

class EarlyStopping:
    '''Class for Early stopping the training if validation loss doesn't improve after a given patience.
    
    Args:
        dir (str): directory to save the model and parameters.
                    Default: 'data/model/'
        patience (int): How long to wait after validation loss improved last time.
                    Default: 10
        verbose (bool): If True, prints a message for each validation loss.
                    Default: False
        checkpoint (bool): If True, saves the model and checkpoint as the performance is getting better. 
                    Default: True.
        torl (float): Minimal change in the monitored quantity to qualify as an improvement.
                    Default: 1e-4
        iter (str): suffix for the name of created model and checkpoint.
                    Default: ""
    '''

    def __init__(self, dir='data/model/', patience=10, verbose=False, checkpoint = True, torl=1e-4, iter = ""):
        '''Initialize EarlyStopping with dir, patience, verbose, checkpoint, torl, iter.'''
        self.dir = dir
        self.patience = patience
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.torl = torl
        self.iter = '_'+iter if iter!="" else iter
        
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Call EarlyStopping instance in each epoch or batch."""
        # If the first epoch, save model.
        if not hasattr(self, "sta_loss"):                   
            self.sta_loss = val_loss
            self.save_checkpoint(val_loss, model)

        # If validation loss drop greater than torlerance, save model.
        elif val_loss < self.sta_loss - self.torl:
            self.save_checkpoint(val_loss, model)
            self.sta_loss = val_loss
            self.counter = 0
        
        # If validation loss does not drop up to the tolerance, count it.
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss drop greater than the torl.'''
        if self.verbose:
            print(f"Validation loss decreased ({self.sta_loss:.8f} --> {val_loss:.8f}). Saving model ...")

        if self.checkpoint:
            torch.save(model.state_dict(), self.dir+f'checkpoint{self.iter}.pt') # The best model parameters up to now.
            torch.save(model, self.dir+f'model{self.iter}.pkl')                  # The best model up to now.