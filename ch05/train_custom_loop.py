from gettext import npgettext
import sys
sys.path.join('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

if __name__ == "__main__":
    batch_size