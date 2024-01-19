import numpy as np

def param_cost(n_layers, dim, order, order_expand, ffw_expand):
    return n_layers * (dim * dim * order * order_expand * 3 + dim * dim * ffw_expand * 2)

def flops_cost(size, kernel_size, layer_cost):
    n_tokens = (size//kernel_size)**2
    return n_tokens * layer_cost


size = 256
kernel = 16
n_layers = 12
dim = 256
order = 4
order_expand = 8
ffw_expand = 4
print('size: {} kernel_size: {} n_layers: {} dim: {} order: {} order_expand: {} ffw_expand: {}'.format(size,
                                                                                                       kernel,
                                                                                                       n_layers,
                                                                                                       dim,
                                                                                                       order,
                                                                                                       order_expand,
                                                                                                       ffw_expand))

pc = param_cost(n_layers, dim, order, order_expand, ffw_expand)
f = flops_cost(size, kernel, pc)
print("params: {}M flops: {}M".format(pc/1000000, f/1000000))