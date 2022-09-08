from torch_geometric.nn import TAGConv

conv_bias = TAGConv(2, 2, bias=True)
conv = TAGConv(2, 2, bias=False)

# excpect all elements of conv_bias.lins to have bias
assert conv_bias.lins[0].bias is not None

# expect all elements of conv.lins to not have bias
assert conv.lins[0].bias is None  # this fails
