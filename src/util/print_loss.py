import os
import time


class LossUtil():
    def __init__(self, log_dir):
        self.log_name = os.path.join(log_dir, 'log_loss.txt')
        with open(self.log_name, "a") as log_loss:
            now = time.strftime("%c")
            log_loss.write('Training Loss (%s)\n' % now)

    def print_losses(self, epoch, losses):
        """
        現在のlossを表示して，log_lossに残す

        """
        message = '(epoch: %d) ' % (epoch)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message

        with open(self.log_name, "a") as log_loss:
            log_loss.write('%s\n' % message)


def print_losses(epoch, losses):
    """
    現在のlossを表示して，log_lossに残す

    """
    message = '(epoch: %d) ' % (epoch)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)  # print the message
