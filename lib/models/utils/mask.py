import torch
import matplotlib.pyplot as plt

def gen_hierarchical_mask(depth=3, seqlen=64, output_seqlen=1):
    step = (seqlen - output_seqlen) // depth        # 21

    hierarchical_mask = torch.zeros((depth, seqlen, seqlen))
    for d in range(depth):
        start, end = step//2 * (d+1), seqlen-(step//2 * (d+1))
        print(start, end)
        hierarchical_mask[d, :, start:end] = 1

    plt.imshow(hierarchical_mask[-1])
    plt.show()
    plt.close()

    return

gen_hierarchical_mask()