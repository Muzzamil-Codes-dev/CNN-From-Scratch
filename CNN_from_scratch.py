import numpy as np
import matplotlib.pyplot as plt
import time


from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to numpy and normalise to [0,1]
Xtr = train_dataset.data.numpy().astype(np.float32) / 255.0   # (60000,28,28)
Ytr = train_dataset.targets.numpy()
Xte = test_dataset.data.numpy().astype(np.float32) / 255.0    # (10000,28,28)
Yte = test_dataset.targets.numpy()

print('train images:', Xtr.shape, ' labels:', Ytr.shape)
print('test  images:', Xte.shape, ' labels:', Yte.shape)

# Show a few examples
fig, ax = plt.subplots(1, 6, figsize=(10, 2))
for i in range(6):
    ax[i].imshow(Xtr[i], cmap='gray')
    ax[i].set_title(f'label={Ytr[i]}')
    ax[i].axis('off')
plt.show()


def ReLU(x):
    return np.maximum(0.0, x)

def softmax(x):
    # subtract max for numerical stability
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)

def cross_entropy(y_pred, y_true):
    # y_true is a one-hot vector (length 10)
    eps = 1e-12
    return -float(np.dot(np.log(y_pred + eps), y_true))


def im2col(x, kH, kW, stride=1):
    """
    x   : (C, H, W)
    out : (C*kH*kW, oH*oW)  -- each column is one receptive field
    """
    C, H, W = x.shape
    oH = (H - kH) // stride + 1
    oW = (W - kW) // stride + 1
    cols = np.zeros((C * kH * kW, oH * oW), dtype=x.dtype)
    col = 0
    for i in range(oH):
        for j in range(oW):
            patch = x[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, oH, oW

def col2im(cols, C, H, W, kH, kW, stride=1):
    """Inverse of im2col: scatter-add columns back into an image of shape (C,H,W)."""
    oH = (H - kH) // stride + 1
    oW = (W - kW) // stride + 1
    x = np.zeros((C, H, W), dtype=cols.dtype)
    col = 0
    for i in range(oH):
        for j in range(oW):
            patch = cols[:, col].reshape(C, kH, kW)
            x[:, i*stride:i*stride+kH, j*stride:j*stride+kW] += patch
            col += 1
    return x


def conv_forward(x, W, b):
    """
    x : (Cin, H, W)
    W : (Cout, Cin, kH, kW)
    b : (Cout,)
    returns y of shape (Cout, oH, oW), plus a cache for backward.
    """
    Cout, Cin, kH, kW = W.shape
    cols, oH, oW = im2col(x, kH, kW)
    Wrow = W.reshape(Cout, -1)
    y = Wrow @ cols + b[:, None]
    return y.reshape(Cout, oH, oW), (x.shape, cols, Wrow, kH, kW, oH, oW)


def maxpool_forward(x, k=2, stride=2):
    """
    x : (C, H, W)
    returns y of shape (C, oH, oW), plus a cache (argmax positions).
    """
    C, H, W = x.shape
    oH = (H - k) // stride + 1
    oW = (W - k) // stride + 1
    y   = np.zeros((C, oH, oW), dtype=x.dtype)
    idx = np.zeros((C, oH, oW, 2), dtype=np.int64)
    for c in range(C):
        for i in range(oH):
            for j in range(oW):
                win = x[c, i*stride:i*stride+k, j*stride:j*stride+k]
                a, b = np.unravel_index(np.argmax(win), win.shape)
                y[c, i, j] = win[a, b]
                idx[c, i, j, 0] = i*stride + a
                idx[c, i, j, 1] = j*stride + b
    return y, (x.shape, idx, k, stride)


def forward(image, params):
    """Run a single image through the network.  Returns prediction and cache."""
    W1, b1, W2, b2, W3, b3 = params

    a1, c_conv1 = conv_forward(image, W1, b1)   # (8,26,26)
    z1 = ReLU(a1)
    p1, c_pool1 = maxpool_forward(z1)            # (8,13,13)

    a2, c_conv2 = conv_forward(p1, W2, b2)       # (16,11,11)
    z2 = ReLU(a2)
    p2, c_pool2 = maxpool_forward(z2)            # (16, 5, 5)

    flat = p2.reshape(-1)                        # 400
    logits = W3 @ flat + b3                      # (10,)
    y = softmax(logits)

    cache = (image, c_conv1, a1, z1, c_pool1,
             c_conv2, a2, z2, c_pool2,
             flat, logits)
    return y, cache


def conv_backward(dY, cache, W):
    x_shape, cols, Wrow, kH, kW, oH, oW = cache
    Cout = W.shape[0]
    dy_flat = dY.reshape(Cout, -1)            # (Cout, oH*oW)
    dWrow = dy_flat @ cols.T                  # (Cout, Cin*kH*kW)
    dW    = dWrow.reshape(W.shape)
    db    = dy_flat.sum(axis=1)               # (Cout,)
    dcols = Wrow.T @ dy_flat                  # (Cin*kH*kW, oH*oW)
    Cin, H, Wd = x_shape
    dx    = col2im(dcols, Cin, H, Wd, kH, kW)
    return dx, dW, db

def maxpool_backward(dY, cache):
    x_shape, idx, k, stride = cache
    dx = np.zeros(x_shape, dtype=dY.dtype)
    C, oH, oW = dY.shape
    for c in range(C):
        for i in range(oH):
            for j in range(oW):
                ii, jj = idx[c, i, j]
                dx[c, ii, jj] += dY[c, i, j]
    return dx

def backward(y_pred, y_true, cache, params):
    W1, b1, W2, b2, W3, b3 = params
    (image, c_conv1, a1, z1, c_pool1,
     c_conv2, a2, z2, c_pool2,
     flat, logits) = cache

    # softmax + CE
    dlogits = y_pred - y_true                  # (10,)
    dW3 = np.outer(dlogits, flat)              # (10, 400)
    db3 = dlogits
    dflat = W3.T @ dlogits                     # (400,)

    # back through pool2 (shape (16,5,5))
    dp2 = dflat.reshape(c_pool2[1].shape[:3])
    dz2 = maxpool_backward(dp2, c_pool2)
    da2 = dz2 * (a2 > 0)                       # ReLU'
    dp1, dW2, db2 = conv_backward(da2, c_conv2, W2)

    dz1 = maxpool_backward(dp1, c_pool1)
    da1 = dz1 * (a1 > 0)
    _, dW1, db1 = conv_backward(da1, c_conv1, W1)

    return [dW1, db1, dW2, db2, dW3, db3]


def init_params(seed=1):
    g = np.random.default_rng(seed)
    W1 = g.standard_normal((8, 1, 3, 3))  * np.sqrt(2.0 / (1*3*3))
    b1 = np.zeros(8)
    W2 = g.standard_normal((16, 8, 3, 3)) * np.sqrt(2.0 / (8*3*3))
    b2 = np.zeros(16)
    W3 = g.standard_normal((10, 16*5*5))  * np.sqrt(2.0 / (16*5*5))
    b3 = np.zeros(10)
    return [W1, b1, W2, b2, W3, b3]


def grad_check():
    params = init_params(seed=42)
    rng = np.random.default_rng(0)
    image = rng.standard_normal((1, 28, 28))
    y_true = np.zeros(10); y_true[3] = 1.0

    y_pred, cache = forward(image, params)
    grads = backward(y_pred, y_true, cache, params)

    h = 1e-5
    tests = [("W1[0,0,1,1]", 0, (0,0,1,1)),
             ("W2[3,2,0,1]", 2, (3,2,0,1)),
             ("W3[2,5]",     4, (2,5)),
             ("b3[7]",       5, (7,))]
    print(f"{'parameter':14s} {'analytic':>14s} {'numerical':>14s} {'rel err':>10s}")
    print('-'*56)
    for name, pi, idx in tests:
        p = params[pi]; g = grads[pi]
        old = p[idx]
        p[idx] = old + h
        L_plus,  _ = forward(image, params); L_plus  = cross_entropy(L_plus,  y_true)
        p[idx] = old - h
        L_minus, _ = forward(image, params); L_minus = cross_entropy(L_minus, y_true)
        p[idx] = old
        num = (L_plus - L_minus) / (2*h)
        ana = g[idx]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        print(f"{name:14s} {ana:>+14.6e} {num:>+14.6e} {rel:>10.2e}")

grad_check()


# Training hyperparameters
N_STEPS = 3000
BATCH   = 8
lr      = 0.01

params = init_params(seed=7)
rng    = np.random.default_rng(123)

loss_history = []
t0 = time.time()
running = 0.0
for step in range(N_STEPS):
    idx_batch = rng.choice(Xtr.shape[0], BATCH, replace=False)
    grads_sum = None
    loss_sum  = 0.0
    for k in idx_batch:
        image  = Xtr[k][None, :, :]                  # (1,28,28)
        y_true = np.zeros(10); y_true[Ytr[k]] = 1.0
        y_pred, cache = forward(image, params)
        loss_sum += cross_entropy(y_pred, y_true)
        g = backward(y_pred, y_true, cache, params)
        if grads_sum is None:
            grads_sum = g
        else:
            for i in range(len(g)):
                grads_sum[i] = grads_sum[i] + g[i]
    # SGD update with average gradient over the mini-batch
    for i in range(len(params)):
        params[i] -= lr * grads_sum[i] / BATCH

    running += loss_sum / BATCH
    if (step + 1) % 100 == 0:
        avg = running / 100
        loss_history.append(avg)
        running = 0.0
        print(f"step {step+1:5d}/{N_STEPS}  avg loss = {avg:.4f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")


plt.figure(figsize=(7,4))
plt.plot(np.arange(1, len(loss_history)+1) * 100, loss_history)
plt.xlabel('training step')
plt.ylabel('cross-entropy loss (per 100 steps)')
plt.title('Training loss')
plt.grid(True)
plt.show()


def predict(image, params):
    y, _ = forward(image, params)
    return int(np.argmax(y))

n_correct = 0
N_TEST = Xte.shape[0]
for k in range(N_TEST):
    if predict(Xte[k][None,:,:], params) == Yte[k]:
        n_correct += 1
acc = n_correct / N_TEST
print(f"Test accuracy: {n_correct}/{N_TEST} = {100*acc:.2f}%")


# Confusion matrix
conf = np.zeros((10,10), dtype=np.int64)
preds = np.zeros(N_TEST, dtype=np.int64)
for k in range(N_TEST):
    p = predict(Xte[k][None,:,:], params)
    preds[k] = p
    conf[Yte[k], p] += 1
print("Confusion matrix (rows = true label, columns = predicted):")
print(conf)
print()
for c in range(10):
    correct = conf[c, c]; total = conf[c].sum()
    print(f"  digit {c}: {correct}/{total} = {100*correct/total:.1f}%")


# Show some correctly-classified and some mis-classified examples
correct_idx   = np.where(preds == Yte)[0][:6]
incorrect_idx = np.where(preds != Yte)[0][:6]

fig, ax = plt.subplots(2, 6, figsize=(11, 4))
for j, k in enumerate(correct_idx):
    ax[0,j].imshow(Xte[k], cmap='gray')
    ax[0,j].set_title(f'✓  true={Yte[k]}, pred={preds[k]}', fontsize=10)
    ax[0,j].axis('off')
for j, k in enumerate(incorrect_idx):
    ax[1,j].imshow(Xte[k], cmap='gray')
    ax[1,j].set_title(f'✗  true={Yte[k]}, pred={preds[k]}', fontsize=10)
    ax[1,j].axis('off')
plt.tight_layout()
plt.show()


W1 = params[0]   # (8, 1, 3, 3)
fig, ax = plt.subplots(1, 8, figsize=(12, 2))
for c in range(8):
    ax[c].imshow(W1[c, 0], cmap='gray')
    ax[c].set_title(f'filter {c}')
    ax[c].axis('off')
plt.suptitle('Learned first-layer 3x3 filters', y=1.05)
plt.show()

