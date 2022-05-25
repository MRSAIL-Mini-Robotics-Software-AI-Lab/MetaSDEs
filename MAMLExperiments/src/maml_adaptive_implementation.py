# -*- coding: utf-8 -*-
"""MAML Adaptive Implementation.ipynb

Automatically generated by Colaboratory.
"""

# from google.colab import drive
# drive.mount('/content/drive')
# %cd drive/MyDrive/Projects/MAML
# !ls

# Commented out IPython magic to ensure Python compatibility.
# %ls

from torch.optim import Adam, SGD
from copy import deepcopy
from torch import autograd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd  # pylint: disable=unused-import
from torch.utils import tensorboard
# import util  # pylint: disable=unused-import
import util
import omniglot

BATCH_SIZE = 16
NUM_WAY = 5
NUM_SUPPORT = 15
NUM_QUERY = 5
NUM_TRAINING_ITERATIONS = 15000
NUM_TRAINING_TASKS = BATCH_SIZE*(NUM_TRAINING_ITERATIONS)
NUM_TEST_TASKS = 600

trainloader = omniglot.get_omniglot_dataloader(
    'train',
    BATCH_SIZE,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    NUM_TRAINING_TASKS
)

validloader = omniglot.get_omniglot_dataloader(
    'val',
    BATCH_SIZE,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    BATCH_SIZE * 4
)

dataloader_test = omniglot.get_omniglot_dataloader(
    'test',
    1,
    NUM_WAY,
    NUM_SUPPORT,
    NUM_QUERY,
    NUM_TEST_TASKS
)

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600


class simpleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            NUM_INPUT_CHANNELS, NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv2 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv3 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv4 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.linear = nn.Linear(NUM_HIDDEN_CHANNELS, num_classes)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv2.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv3.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv4.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = torch.mean(x, dim=[2, 3])

        x = self.linear.forward(x)

        return x


class MAML:
    def __init__(self, inner_steps: int,
                 inner_lr: float,
                 outer_lr: float,
                 model_class: type,
                 *model_args: any):
        '''
        initiates the MAML class

        Parameters:
        -----------
        inner_steps: int - number of gradient descent update steps
        inner_lr: float - learning rate of the inner gradient step with the support dset
        outer_lr: float - learning rate of the outer gradient step with the query dset
        model_class: type - the model used for metalearning
        lossfn: callable - loss function
        '''
        self.inner_steps = inner_steps
        self.inner_lr = nn.Parameter(torch.Tensor([inner_lr]))
        self.outer_lr = nn.Parameter(torch.Tensor([outer_lr]))
        self.model_class = model_class

    def __cloneModule(self, target: nn.Module, instance: nn.Module):
        '''
        clones the input module in a way to be present in the computational graph

        Parameters:
        -----------
        target: torch.nn - the target model to clone
        instance: torch.nn - instance of the same class to clone into
        '''
        # clone parameters of the current level
        for param_key in target._parameters:
            if target._parameters[param_key] is not None:
                instance._parameters[param_key] = target._parameters[param_key].clone(
                )

        # recursively go to the other children parameters and do the same
        for target_child, instance_child in zip(target.children(), instance.children()):
            self.__cloneModule(target_child, instance_child)

    def __innerLoop(self, model: nn.Module, criterion: callable, images, labels=None,):
        '''
        performs a GD step update on a given model using the given loss fn

        Parameters:
        -----------
        model: nn.Module - cloned model
        loss: callable - loss function
        '''
        # perform gradient updates
        model.train()
        device = labels.device

        for _ in range(self.inner_steps):
            outs = model(images)
            if labels is None:
                loss = criterion(images, outs)
            else:
                loss = criterion(outs, labels)

            grads = autograd.grad(
                loss, [*model.parameters()], allow_unused=True, create_graph=True)
            for param, grad in zip(model.parameters(), grads):
                param.data -= self.inner_lr.to(device)*grad

        model.eval()
        with torch.no_grad():
            outs = model(images)
            loss = criterion(outs, labels)

            accuracy_inner = None

            if labels is not None:
                accuracy_inner = util.score(outs, labels)

        # @NOTE reset the finetune model to thre training mode
        model.train()

        return model, loss.item(), accuracy_inner

    def train(self,
              trainloader,
              validloader,
              criterion,
              print_every=25,
              validate_every=50,
              ):
        '''
        trains the model using MAML finetuning

        Parameters:
        -----------
        trainloader: list - contains Tensors of task batches
        validloader: list - contains Tensors of task batches 
        criterion: callable - loss function
        epochs: int - number of training iterations

        Returns:
        --------
        sli_arr: arr - support loss mean for each epoch for the inner loop
        slo_arr: arr - support loss mean for each epoch for the outerloop
        qli_arr: arr - query loss mean for each epoch for the outerloop
        '''
        # presets
        fineTuneModel = self.model_class()
        targetModel = self.model_class()
        optimizer = SGD([*targetModel.parameters()], lr=self.outer_lr.item())

        # pre adaptation support batch loss and accuracy memo
        pre_adapt_support_batch_acc_memo = []
        pre_adapt_query_batch_loss_memo = []
        # pre adaptation query batch loss and accuracy memo
        pre_adapt_query_batch_acc_memo = []
        pre_adapt_query_batch_loss_memo = []
        # post adaptation query batch loss and accuracy memo
        post_adapt_query_batch_acc_memo = []
        post_adapt_query_batch_loss_memo = []

        # put model on active device
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        print("[Message] Training on", device)
        fineTuneModel = fineTuneModel.to(device)
        targetModel = targetModel.to(device)

        for i, task_batch in tqdm(enumerate(trainloader)):

            # pre adaptation support batch losses and accs (for each batch)
            pre_adapt_support_batch_acc = []
            pre_adapt_support_batch_loss = []
            # pre adaptation query batch losses and accs (for each batch)
            pre_adapt_query_batch_acc = []
            pre_adapt_query_batch_loss = []
            # post adaptation query batch losses and accs (for each batch)
            post_adaptation_query_batch_loss = []
            post_adaptation_query_batch_acc = []

            for images_s,  labels_s, images_q, labels_q in task_batch:

                # clone models
                self.__cloneModule(targetModel, fineTuneModel)

                # putting data in device
                targetModel.train()
                fineTuneModel.train()

                images_s, labels_s = images_s.to(device), labels_s.to(device)
                images_q, labels_q = images_q.to(device), labels_q.to(device)

                _, inner_support_loss, inner_support_accuracies = self.__innerLoop(
                    fineTuneModel, criterion, images_s, labels_s)

                # outerloop and update
                outs = fineTuneModel(images_q)
                loss = criterion(outs, labels_q)

                # calculate pre-adaptation scores
                pre_adapt_support_batch_acc.append(inner_support_accuracies)
                pre_adapt_support_batch_loss.append(inner_support_loss)
                pre_adapt_query_batch_acc.append(util.score(outs, labels_q))
                pre_adapt_query_batch_loss.append(loss.item())

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('post', next(iter(targetModel.parameters())))

                # calculate post-adaptation scores
                targetModel.eval()

                self.__cloneModule(targetModel, fineTuneModel)

                self.__innerLoop(fineTuneModel, criterion,
                                 images_q, labels_q)
                outs = fineTuneModel(images_q)
                loss = criterion(outs, labels_q)
                post_adaptation_query_batch_acc.append(
                    util.score(outs, labels_q))
                post_adaptation_query_batch_loss.append(loss.item())

                # @TODO take the best model according to the query error minimization

            if (i % print_every) == 0:
                # inner scores
                pre_adapt_support_batch_acc = np.mean(
                    pre_adapt_support_batch_acc)
                pre_adapt_support_batch_loss = np.mean(
                    pre_adapt_support_batch_loss)
                pre_adapt_support_batch_acc_memo.append(
                    pre_adapt_support_batch_acc)
                pre_adapt_query_batch_acc_memo.append(
                    pre_adapt_support_batch_loss)

                # pre adaptation outer scores
                pre_adapt_query_batch_acc = np.mean(pre_adapt_query_batch_acc)
                pre_adapt_query_batch_loss = np.mean(
                    pre_adapt_query_batch_loss)
                pre_adapt_query_batch_loss_memo.append(
                    pre_adapt_query_batch_acc)
                pre_adapt_query_batch_loss_memo.append(
                    pre_adapt_query_batch_loss)

                # post adaptations outer scores
                post_adaptation_query_batch_acc = np.mean(
                    post_adaptation_query_batch_acc)
                post_adaptation_query_batch_loss = np.mean(
                    post_adaptation_query_batch_loss)
                post_adapt_query_batch_loss_memo.append(
                    post_adaptation_query_batch_acc)
                post_adapt_query_batch_acc_memo.append(
                    post_adaptation_query_batch_loss)

                # verbose message
                message = f'''
                [+ Training Log @ iter{i}]----------------------------
                [Pre-Adaptation]
                -Pre-support accuracy score =\t{pre_adapt_support_batch_acc}
                -Pre-support loss=\t{pre_adapt_support_batch_loss}
                [Pre-Adaptation Scores]
                -outer query accuracy score =\t{pre_adapt_query_batch_acc}
                -outer query loss =\t{pre_adapt_query_batch_loss}
                ||-----[After Meta training]--->>
                [Post-Adaptation Scores]
                -outer query accuracy score =\t{post_adaptation_query_batch_acc}
                -outer query loss =\t{post_adaptation_query_batch_loss}
                '''
                print(message)

        self.targetModel = targetModel
        return targetModel

    def validateModel(self, model, validloader, criterion, valid_batch_verbose=True):
        '''
        validates input data

        Parameters
        ----------
        model: model object -  meta trained model
        validloader: iter - torch validation loader
        criterion: callable - criterion function

        Returns
        ------
        losses: dict - dictionary of losses (averaged on all batches)  {
            'valid_loss_inner': float,
            'valid_support_loss_before_inner':float,
            'valid_support_loss_after_inner':float,
            'valid_query_loss_before_inner': float,
            'valid_query_loss_after_inner': float
        }
        '''
        valid_model = deepcopy(model)
        valid_model.train()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        losses = {
            'valid_loss_inner': 0,
            'valid_support_loss_before_inner': 0,
            'valid_support_loss_after_inner': 0,
            'valid_query_loss_before_inner': 0,
            'valid_query_loss_after_inner': 0
        }

        for data in validloader:
            for images_s,  labels_s, images_q, labels_q in data:

                images_s, labels_s = images_s.to(device), labels_s.to(device)
                images_q, labels_q = images_q.to(device), labels_q.to(device)

                # valid losses before inner
                valid_model.eval()
                with torch.no_grad():
                    losses['valid_support_loss_before_inner'] += criterion(
                        valid_model(images_s), labels_s)
                    losses['valid_query_loss_before_inner'] += criterion(
                        valid_model(images_q), labels_q)

                # training with the support images
                valid_model.train()

                _, loss_s_inner = self.__innerLoop(
                    valid_model, criterion, images_s, labels_s)
                losses['valid_loss_inner'] += loss_s_inner

                # valid losses after training
                valid_model.eval()
                with torch.no_grad():
                    losses['valid_support_loss_after_inner'] += criterion(
                        valid_model(images_s), labels_s)
                    losses['valid_query_loss_after_inner'] += criterion(
                        valid_model(images_q), labels_q)

        losses['valid_loss_inner'] /= len(validloader)
        losses['valid_support_loss_before_inner'] /= (
            len(validloader)*images_s.shape[0])
        losses['valid_query_loss_before_inner'] /= (
            len(validloader)*images_q.shape[0])
        losses['valid_support_loss_after_inner'] /= (
            len(validloader)*images_s.shape[0])
        losses['valid_query_loss_after_inner'] /= (
            len(validloader)*images_q.shape[0])

        if valid_batch_verbose:
            message = f'''
            ---Validation Batch Results---
            valid loss inner:\t\t{losses['valid_loss_inner']}
            support loss outer before training:\t\t{losses['valid_support_loss_before_inner']}
            support loss outer after training:\t\t{losses['valid_query_loss_before_inner']}
            query loss before training:\t\t{losses['valid_support_loss_after_inner']}
            query loss after training:\t\t{losses['valid_query_loss_after_inner']}
            '''
            print(message)

        return losses


if __name__ == "__main__":
    model = MAML(2, 1, 1, lambda: simpleNet(5))
    # model = MAML(2, 1e-3, 1, lambda: simpleNet(5))
    model.train(trainloader=trainloader, validloader=validloader,
                criterion=nn.CrossEntropyLoss(), print_every=50)
    torch.save(model.targetModel.state_dict(), 'targetModel.pth')
