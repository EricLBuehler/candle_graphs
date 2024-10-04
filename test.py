import torch

N, D_in = 32, 32

# Placeholders used for capture
static_input = torch.randn(N, D_in, device='cuda')

# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        y_pred = static_input @ static_input
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y_pred = static_input @ static_input

print(g)
data = torch.ones(N, D_in, device='cuda')
static_input.copy_(data)

g.replay()

print(static_y_pred)
