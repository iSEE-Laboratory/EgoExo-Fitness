import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return
        # if torch.cuda.is_available():
        #     with torch.cuda.stream(self.stream):
        #         self.next_sample = self.next_sample.cuda(non_blocking=True)

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(self.stream)
        next_sample = self.next_sample
        self.preload()
        return next_sample