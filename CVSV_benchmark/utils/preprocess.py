import torch
def frames_preprocess(frames,flipped=False):

    bs, c, h, w, num_clip = frames.size()
    if flipped:
        frames = torch.flip(frames, dims=[-1])
    frames = frames.permute(0, 4, 1, 2, 3)
    frames = frames.reshape(-1, c, h, w)

    return frames

