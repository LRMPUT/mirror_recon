import shutil

from torch.utils.tensorboard import SummaryWriter

from graspable.misc import *


class Logs:
    def __init__(self, path, clear=True):
        self._path = path
        self._writer = None
        self.create_writer(clear)
        pass

    def create_writer(self, clear):
        def _clear_logs(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        if os.path.exists(self._path):
            if clear:
                _clear_logs(self._path)
        else:
            makedirs(self._path)

        self._writer = SummaryWriter(self._path)

    def update(self, **kwargs):
        pass

    def close(self):
        self._writer.close()
