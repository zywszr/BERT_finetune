# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import torch
import torch.nn as nn
from progress.bar import Bar
from sklearn.cluster import KMeans


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(-1), torch.from_numpy(labels).int().cuda()

def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i in range(len(centroids)):
        weight[labels == i] = centroids[i]
    return weight

def quantize_model(model, quantize_index, quantize_bits, recorder, max_iter=50, mode='cpu', quantize_bias=False,
                   centroids_init='k-means++'):
    assert len(quantize_index) == len(quantize_bits), \
        'You should provide the same number of bit setting as layer list!'
    quantize_layer_bit_dict = {n: b for n, b in zip(quantize_index, quantize_bits)}

    bar = Bar('KMeans:', max=len(quantize_index))
    j = 0
    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:  # if -1, do not quantize
            continue
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b
        # quantize weight
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            if mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[0], init=centroids_init, max_iter=max_iter)
            else:
                raise NotImplementedError
            recorder.__setattr__('centroidsw%d' % j, centroids)
            recorder.__setattr__('labelsw%d' % j, labels)
            for no in range(len(centroids)):
                layer.weight.data[labels==no] = centroids[no]
        # quantize bias
        if hasattr(layer, 'bias') and quantize_bias:
            w = layer.bias.data
            if mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[1], init=centroids_init, max_iter=max_iter)
            else:
                raise NotImplementedError
            recorder.__setattr__('centroidsb%d' % j, centroids)
            recorder.__setattr__('labelsb%d' % j, labels)
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            layer.bias.data = w_q.float()

        bar.suffix = ' id: {id:} | bit: {bit:}'.format(id=i, bit=n_bit[0])
        bar.next()
        j = j + 1
    bar.finish()
    return recorder


def update_quan_bert(model, quantizable_idx, recorder, num):
    j = 0
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            # if type(layer) == nn.Linear:
            #     print(layer.weight.grad)
            continue
        centroids = torch.zeros_like(eval('recorder.centroidsw%d' % j)).float().cuda()
        labels = eval('recorder.labelsw%d' % j)
        # with open('/slfs1/users/zhz73/BERT/result/networkb%d.txt' % num, 'a') as f:
        #     f.write(str(i) + '\n')
        #     f.write(str(centroids) + '\n')
        #     # f.write(str(labels.tolist()) + '\n')
        #     f.write('\n')
        for no in range(len(centroids)):
            centroids[no] = layer.weight.data[labels==no].mean().item()
            layer.weight.data[labels==no] = centroids[no]
        recorder.__setattr__('centroidsw%d' % j, centroids)
        recorder.__setattr__('labelsw%d' % j, labels)
        # with open('/slfs1/users/zhz73/BERT/result/network%d.txt' % num, 'a') as f:
        #     f.write(str(i) + '\n')
        #     f.write(str(centroids) + '\n')
        #     # f.write(str(labels.tolist()) + '\n')
        #     f.write('\n')
        j = j + 1
    return model, recorder
    #! don't apply to bias

