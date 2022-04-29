# -*- coding: utf-8 -*-


import torch
import torch.autograd as autograd


def logsumexp(matrix, dim):
    print(matrix.size())
    t1 = torch.exp(matrix)
    t2 = t1.sum(dim)
    t2.register_hook(lambda x: print('t2 ' + str(x)))
    t2.register_hook(lambda x: x.masked_fill_(torch.isnan(x)+torch.isinf(x), 0))
    ret = torch.log(t2)
    #ret.sum().backward()
    #exit()
    
    #ret.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
    ret.register_hook(lambda x: print('lalala' + str(x)))
    return ret


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.
    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.
    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def pad(tensors, padding_value=0, total_length=None):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def tarjan(sequence):
    sequence[0] = -1
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


@torch.enable_grad()
def crf(scores, mask, target=None, partial=False):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    training = scores.requires_grad
    # always enable the gradient computation of scores
    # in order for the computation of marginal probs
    s_i, s_c = inside(scores.requires_grad_(), mask)
    logZ = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    # marginal probs are used for decoding, and can be computed by
    # combining the inside algorithm and autograd mechanism
    # instead of the entire inside-outside process
    probs, = autograd.grad(logZ, scores, retain_graph=training)

    if target is None:
        return probs
    # the second inside process is needed if use partial annotation
    if partial:
        s_i, s_c = inside(scores, mask, target)
        score = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    else:
        score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
    loss = logZ - score

    return loss, probs


def insideg(scores, mask, cands=None):
    # the end position of each sentence in a batch
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    # [seq_len, seq_len, batch_size]
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)

    # set the scores of arcs excluded by cands to -inf
    if cands is not None:
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
        cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
        cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
        cands = cands.permute(2, 1, 0) & mask
        scores = scores.masked_fill(~cands, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w

        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size]
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        #if ilr.requires_grad:
            #ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
        # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
        # fill the w-th diagonal of the lower triangular part of s_i
        # with I(j->i) of n spans
        s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
        # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
        # fill the w-th diagonal of the upper triangular part of s_i
        # with I(i->j) of n spans
        s_i.diagonal(w).copy_(ir + scores.diagonal(w))

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        #cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        #cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')
    return s_c[0].gather(0, lens.unsqueeze(0))
    #return s_i, s_c


def inside2(scores, mask, cands=None):
    # the end position of each sentence in a batch
    lens = mask.sum(1)
    s_arc, s_sib = scores
    batch_size, seq_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0)
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0)
    s_i = torch.full_like(s_arc, float('-inf'))
    s_s = torch.full_like(s_arc, float('-inf'))
    s_c = torch.full_like(s_arc, float('-inf'))
    s_c.diagonal().fill_(0)

    # set the scores of arcs excluded by cands to -inf
    if cands is not None:
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
        cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
        cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
        cands = cands.permute(2, 1, 0) & mask
        s_arc = s_arc.masked_fill(~cands, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
        #                  exp(C(j->j) + C(i->j-1)))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
        if il.requires_grad:
            il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        il = il.permute(2, 0, 1).logsumexp(-1)
        s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
        # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
        #                  exp(C(i->i) + C(j->i+1)))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
        ir[0] = float('-inf')
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        if ir.requires_grad:
            ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        ir = ir.permute(2, 0, 1).logsumexp(-1)
        s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if slr.requires_grad:
            slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        slr = slr.permute(2, 0, 1).logsumexp(-1)
        # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr)
        # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr)

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    return s_c[0].gather(0, lens.unsqueeze(0))



def inside2g(scores, mask, cands=None):
    # the end position of each sentence in a batch
    lens = mask.sum(1)
    s_arc, s_sib = scores
    batch_size, seq_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0)
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0)
    s_i = torch.full_like(s_arc, float('-inf'))
    s_s = torch.full_like(s_arc, float('-inf'))
    s_c = torch.full_like(s_arc, float('-inf'))
    s_c.diagonal().fill_(0)

    # set the scores of arcs excluded by cands to -inf
    if cands is not None:
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
        cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
        cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
        cands = cands.permute(2, 1, 0) & mask
        s_arc = s_arc.masked_fill(~cands, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
        #                  exp(C(j->j) + C(i->j-1)))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
        if il.requires_grad:
            il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        il = il.permute(2, 0, 1).logsumexp(-1)
        s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
        # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
        #                  exp(C(i->i) + C(j->i+1)))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
        ir[0] = float('-inf')
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        if ir.requires_grad:
            ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        ir = ir.permute(2, 0, 1).logsumexp(-1)
        s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if slr.requires_grad:
            slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        slr = slr.permute(2, 0, 1).logsumexp(-1)
        # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr)
        # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr)

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        #cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        #cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    return s_c[0].gather(0, lens.unsqueeze(0))



def inside(scores, mask, cands=None):
    # the end position of each sentence in a batch
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    # [seq_len, seq_len, batch_size]
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)

    # set the scores of arcs excluded by cands to -inf
    if cands is not None:
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
        cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
        cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
        cands = cands.permute(2, 1, 0) & mask
        scores = scores.masked_fill(~cands, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w

        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size]
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if ilr.requires_grad:
            ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
        # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
        # fill the w-th diagonal of the lower triangular part of s_i
        # with I(j->i) of n spans
        s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
        # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
        # fill the w-th diagonal of the upper triangular part of s_i
        # with I(i->j) of n spans
        s_i.diagonal(w).copy_(ir + scores.diagonal(w))

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')
    return s_c[0].gather(0, lens.unsqueeze(0))
    #return s_i, s_c

def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        il = ir = ilr.permute(2, 0, 1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span + scores.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span + scores.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    preds = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        preds.append(heads.to(mask.device))

    return pad(preds, total_length=seq_len).to(mask.device)
