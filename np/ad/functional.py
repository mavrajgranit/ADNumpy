import numpy as np


def conv1d(xv, kv, kx, ky, channel, stride):
    sx = stride[0]
    sy = stride[1]
    dep = xv.ndim == 3
    nx = int((xv.shape[-1] - kx) / sx + 1)
    ny = int((xv.shape[-2] - ky) / sy + 1)
    amount = ky * kx

    if dep:
        depth = xv.shape[-3]
        nv_s = (depth, amount, nx * ny)
        x_rs = (depth, amount)
        res_rs = (depth, ny * channel, nx)
    else:
        depth = 1
        nv_s = (amount, nx * ny)
        x_rs = amount
        res_rs = (ny * channel, nx)
    nv = np.empty(nv_s)

    i = 0
    for y in range(0, xv.shape[-2]-ky+1, sy):
        for x in range(0, xv.shape[-1]-kx+1, sx):
            if dep:
                nv[:, :, i] = xv[:, y:y + ky, x:x + kx].reshape(x_rs)
            else:
                nv[:, i] = xv[y:y + ky, x:x + kx].reshape(x_rs)
            i += 1

    res = kv @ nv
    return res.reshape(res_rs), res.shape, nv


# only correct in the case of conv1d backward TODO FIX
def conv1d_transpose_(xv, kv, kx, ky, channel, stride):
    gs = xv.shape
    y_amount = int(gs[-2] / channel)
    amount = ky * kx

    sx = stride[0]
    sy = stride[1]
    dep = xv.ndim == 3

    if dep:
        depth = xv.shape[-3]
        xt_s = (depth, channel, y_amount + (ky - 1) * 2, gs[-1] + (kx - 1) * 2)
        xv_rs = (depth, channel, y_amount, gs[-1])
        x_rs = (depth, amount, 1)
    else:
        depth = 1
        xt_s = (channel, y_amount + (ky - 1) * 2 + sy - 1, gs[-1] + (kx - 1) * 2 + sx - 1)
        xv_rs = (channel, y_amount, gs[-1])
        x_rs = amount

    x_t = np.zeros(xt_s)

    """upx = xv.shape[-1] if kx == 1 else -kx + 1
    upy = xv.shape[-2] if ky == 1 else -ky + 1
    if dep:
        x_t[:, :, ky - 1:upy, kx - 1:upx] = xv.reshape(xv_rs)
    else:
        x_t[:, ky-1:upy, kx-1:upx] = xv.reshape(xv_rs)"""
    for y in range(xv.shape[-2]):
        for x in range(xv.shape[-1]):
            if dep:
                x_t[:, int(y / y_amount), ky - 1 + (y % y_amount) * sy, kx - 1 + x * sx] = xv[:, y, x]
            else:
                x_t[int(y / y_amount), ky - 1 + (y % y_amount) * sy, kx - 1 + x * sx] = xv[y, x]

    nx = (x_t.shape[-1] - kx) + 1
    ny = (x_t.shape[-2] - ky) + 1

    res = np.zeros((depth, ny, nx)) if dep else np.zeros((ny, nx))

    for y in range(ny):
        for x in range(nx):
            for c in range(channel):
                if dep:
                    res[:, y, None, x, None] += kv[np.newaxis, c, :] @ x_t[:, c, y:y + ky, x:x + kx].reshape(x_rs)
                else:
                    res[y, x] += kv[c, :] @ x_t[c, y:y + ky, x:x + kx].reshape(x_rs)
    return res


def conv1d_transpose(xv, kv, kx, ky, channel, stride):
    sx, sy = stride
    xy = xv.shape[-2]
    xx = xv.shape[-1]
    dep = xv.ndim == 3

    if dep:
        xt_s = (xv.shape[-3], xy + (ky - 1) * 2 + (sy - 1) * (xy - 1), xx + (kx - 1) * 2 + (sx - 1) * (xx - 1))
    else:
        xt_s = (xy + (ky - 1) * 2 + (sy - 1) * (xy - 1), xx + (kx - 1) * 2 + (sx - 1) * (xx - 1))

    x_t = np.zeros(xt_s)
    for y in range(xy):
        for x in range(xx):
            if dep:
                x_t[:, y * sy + ky - 1, x * sx + kx - 1] = xv[:, y, x]
            else:
                x_t[y * sy + ky - 1, x * sx + kx - 1] = xv[y, x]
    return conv1d(x_t, kv, kx, ky, channel, (1, 1))


def conv2d(xv, kv, kx, ky, kz, channel, stride):
    sx, sy, sz = stride
    nx = int((xv.shape[-1] - kx) / sx) + 1
    ny = int((xv.shape[-2] - ky) / sy) + 1

    dep4 = xv.ndim == 4
    depth4 = xv.shape[-4] if dep4 else 1

    dep3 = xv.ndim == 3
    depth3 = xv.shape[-3] if dep3 or dep4 else 1

    nz = int((depth3 - kz) / sz) + 1

    amount = kx * ky * kz
    out_amount = nx * ny * nz

    if dep4:
        nv_s = (depth4, amount, out_amount)
        x_rs = (depth4, amount)
        res_rs = (depth4, nz * channel, ny, nx)
    elif dep3:
        nv_s = (amount, out_amount)
        x_rs = amount
        res_rs = (nz * channel, ny, nx)
    else:
        nv_s = (amount, out_amount)
        x_rs = amount
        res_rs = (nz * channel, ny, nx)

    nv = np.empty(nv_s)
    i = 0
    for z in range(0, depth3 - kz + 1, sz):
        for y in range(0, xv.shape[-2] - ky + 1, sy):
            for x in range(0, xv.shape[-1] - kx + 1, sx):
                if dep4:
                    nv[:, :, i] = xv[:, z:z + kz, y:y + ky, x:x + kx].reshape(x_rs)
                elif dep3:
                    nv[:, i] = xv[z:z + kz, y:y + ky, x:x + kx].reshape(x_rs)
                else:
                    nv[:, i] = xv[y:y + ky, x:x + kx].reshape(x_rs)
                i += 1

    res = kv @ nv
    return res.reshape(res_rs), res.shape, nv


def conv2d_transpose(xv, kv, kx, ky, kz, channel, stride):
    sx, sy, sz = stride
    xy = xv.shape[-2]
    xx = xv.shape[-1]

    dep4 = xv.ndim == 4
    depth4 = xv.shape[-4] if dep4 else 1

    dep3 = xv.ndim == 3
    xz = xv.shape[-3] if dep3 or dep4 else 1

    if dep4:
        xt_s = (depth4, xz + (kz - 1) * 2 + (sz - 1) * (xz - 1), xy + (ky - 1) * 2 + (sy - 1) * (xy - 1),
                xx + (kx - 1) * 2 + (sx - 1) * (xx - 1))
    elif dep3:
        xt_s = (xz + (kz - 1) * 2 + (sz - 1) * (xz - 1), xy + (ky - 1) * 2 + (sy - 1) * (xy - 1),
                xx + (kx - 1) * 2 + (sx - 1) * (xx - 1))
    else:
        xt_s = (xy + (ky - 1) * 2 + (sy - 1) * (xy - 1), xx + (kx - 1) * 2 + (sx - 1) * (xx - 1))

    x_t = np.zeros(xt_s)
    for z in range(xz):
        for y in range(xy):
            for x in range(xx):
                if dep4:
                    x_t[:, z * sz + kz - 1, y * sy + ky - 1, x * sx + kx - 1] = xv[:, z, y, x]
                elif dep3:
                    x_t[z * sz + kz - 1, y * sy + ky - 1, x * sx + kx - 1] = xv[z, y, x]
                else:
                    x_t[y * sy + ky - 1, x * sx + kx - 1] = xv[y, x]
    return conv2d(x_t, kv, kx, ky, kz, channel, (1, 1, 1))

