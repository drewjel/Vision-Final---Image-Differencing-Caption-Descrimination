import math
import time

class Logger(object):
    def __init__(self, total_batches, is_training=True, log_interval=30):
        self.total_batches = total_batches
        self.is_training = is_training
        self.log_interval = log_interval if is_training else total_batches
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.total_recall1 = 0
        self.total_acc = 0
        self.start_time = time.time()

    def set_values(self, loss, recall1, acc):
        self.total_loss += loss
        self.total_recall1 += recall1
        self.total_acc += acc

    def print_log(self, batch=0, epoch=0, lr=0):
        if not self.is_training or batch % self.log_interval == self.log_interval-1:
            cur_loss = self.total_loss / self.log_interval
            cur_recall = self.total_recall1 / self.log_interval
            cur_acc = self.total_acc / self.log_interval
            elapsed = time.time() - self.start_time
            log = 'train| epoch {:3d} | {:4d}/'.format(epoch, batch) if self.is_training else 'eval'
            log += '{:4d} batches | '.format(self.total_batches)
            log += 'lr {:02.5f} | '.format(lr) if self.is_training else ''
            log += ('ms/batch {:5.2f} | loss {:5.2f} | ppl {:4.2f} | '
                'recall1 {:02.2f} | acc {:02.2f}'.format(
                elapsed * 1000 / self.log_interval,
                cur_loss, math.exp(cur_loss), cur_recall, cur_acc))
            print(log)
            self.reset()