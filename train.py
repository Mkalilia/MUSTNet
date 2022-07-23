from __future__ import absolute_import, division, print_function

from trainer import Trainer
from option.options import MustNetOptions

options = MustNetOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
