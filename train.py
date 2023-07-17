import datetime
import time

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import seed_everything
from util.visualizer import Visualizer

if __name__ == '__main__':
  opt = TrainOptions().parse()   # get training options
  # create a dataset given opt.dataset_mode and other options
  dataset = create_dataset(opt)
  dataset_size = len(dataset)    # get the number of images in the dataset.
  batch_size = opt.batch_size
  print('The number of training images = %d' % dataset_size)

  # create a model given opt.model and other options
  model = create_model(opt)
  # regular setup: load and print networks; create schedulers
  model.setup(opt)
  model.set_device()             # set device to cuda or cpu
  # create a visualizer that display/save images and plots
  visualizer = Visualizer(opt)
  total_iters = 0                # the total number of training iterations

  if opt.pretrain:
    data_size = 0
    for i in range(opt.pretrain_epoch):
      running_loss = 0
      result = []
      for data in dataset:  # inner loop within one epoch
        data_size += batch_size
        model.set_pretrain_input(data)
        acc, loss = model.pretrain()   # start pretrain
        running_loss += loss
        result.append(acc)
      print("[pretrain] Epoch: {}, Loss: {:.3f} Acc: {:.3f}".format(
          i, running_loss/data_size, np.mean(result)))

  model.train()                  # change to train mode
  seed_everything(seed=opt.seed)
  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
  for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    visualizer.print_message(datetime.datetime.now())
    epoch_start_time = time.time()     # timer for entire epoch
    iter_data_time = time.time()       # timer for data loading per iteration
    # the number of training iterations in current epoch, reset to 0 every epoch
    epoch_iter = 0
    for i, data in enumerate(dataset):  # inner loop within one epoch
      iter_start_time = time.time()  # timer for computation per iteration
      if total_iters % opt.print_freq == 0:
        t_data = iter_start_time - iter_data_time
      visualizer.reset()
      total_iters += batch_size
      epoch_iter += batch_size
      # unpack data from dataset and apply preprocessing
      model.set_input(data)
      # calculate loss functions, get gradients, update network weights
      model.optimize_parameters()

      t_comp = (time.time() - iter_start_time) / batch_size
      visualizer.set_scores(epoch, epoch_iter, float(
          epoch_iter) / dataset_size, model.get_current_scores(), t_comp, t_data)

      if total_iters % opt.print_freq == 0:        # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / batch_size
        visualizer.plot_current_losses(epoch, epoch_iter, float(
            epoch_iter) / dataset_size, losses, t_comp, t_data)

      if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
        visualizer.print_message(
            'saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)

      iter_data_time = time.time()

      if (epoch_iter == len(dataset) or total_iters % opt.eval_step_freq == 0):
        model.compute_visuals()
        t_comp = (time.time() - iter_start_time) / batch_size
        visualizer.display_current_results(
            model.get_current_visuals(), epoch, total_iters)
        if opt.monitor_gnorm:
          gnorms = model.get_cuurent_gnorms()
          visualizer.plot_current_gnorms(epoch, epoch_iter, float(
              epoch_iter) / dataset_size, gnorms, t_comp, t_data)
    visualizer.plot_current_scores(epoch)

    if epoch % opt.save_epoch_freq == 0:             # cache our model every <save_epoch_freq> epochs
      visualizer.print_message(
          'saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
      model.save_networks('latest')
      model.save_networks(epoch)

    visualizer.print_message('End of epoch %d / %d \t Time Taken: %d sec' %
                             (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    # update learning rates at the end of every epoch.
    model.update_learning_rate()
    # update weight of constraint at the end of every epoch.
    model.update_lambda_weight(epoch)
