PRIM = 0x11d
EXP = [0] * 512
LOG = [0] * 256

def _init_tables():
    x = 1
    for i in range(255):
        EXP[i] = x
        LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    for i in range(255, 512):
        EXP[i] = EXP[i - 255]

_init_tables()

def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("GF divide by zero")
    return 0 if a == 0 else EXP[(LOG[a] + 255 - LOG[b]) % 255]

def gf_poly_eval(poly, x):
    val = poly[0]
    for c in poly[1:]:
        val = gf_mul(val, x) ^ c
    return val

def rs_generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        tmp = [0] * (len(g) + 1)
        for idx, val in enumerate(g):
            tmp[idx] ^= val
            tmp[idx + 1] ^= gf_mul(val, EXP[i])
        g = tmp
    return g

def rs_encode_block(msg_bytes, nsym):
    gen = rs_generator_poly(nsym)
    res = list(msg_bytes) + [0] * nsym
    for i in range(len(msg_bytes)):
        coef = res[i]
        if coef != 0:
            for j in range(1, len(gen)):
                res[i + j] ^= gf_mul(gen[j], coef)
    return bytes(list(msg_bytes) + res[len(msg_bytes):])

def rs_decode_block(received_bytes, nsym):
    rec = list(received_bytes)
    n = len(rec)
    
    synd = [gf_poly_eval(rec, EXP[i]) for i in range(nsym)]
    if max(synd) == 0:
        return bytes(rec[:-nsym])

    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for k in range(nsym):
        d = synd[k]
        for i in range(1, L + 1):
            d ^= gf_mul(C[i], synd[k - i])
        if d == 0:
            m += 1
        elif 2 * L <= k:
            T = list(C)
            scale = gf_div(d, b)
            pad = [0] * m + [gf_mul(val, scale) for val in B]
            while len(C) < len(pad):
                C.append(0)
            for i in range(len(pad)):
                C[i] ^= pad[i]
            L = k + 1 - L
            B = T
            b = d
            m = 1
        else:
            scale = gf_div(d, b)
            pad = [0] * m + [gf_mul(val, scale) for val in B]
            while len(C) < len(pad):
                C.append(0)
            for i in range(len(pad)):
                C[i] ^= pad[i]
            m += 1

    err_pos = []
    for i in range(n):
        val = 0
        p = 1
        step = EXP[(255 - (n - 1 - i)) % 255]
        for c in C:
            val ^= gf_mul(c, p)
            p = gf_mul(p, step)
        if val == 0:
            err_pos.append(i)

    if len(err_pos) > nsym // 2 or len(err_pos) == 0:
        return None

    num_errs = len(err_pos)
    A = [[0] * num_errs for _ in range(num_errs)]
    B_vec = [synd[k] for k in range(num_errs)]
    for r, pos in enumerate(err_pos):
        X_r = EXP[(n - 1 - pos) % 255]
        term = 1
        for k in range(num_errs):
            A[k][r] = term
            term = gf_mul(term, X_r)

    for i in range(num_errs):
        pivot = -1
        for row in range(i, num_errs):
            if A[row][i] != 0:
                pivot = row
                break
        if pivot == -1:
            return None
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            B_vec[i], B_vec[pivot] = B_vec[pivot], B_vec[i]
        inv = gf_div(1, A[i][i])
        for col in range(i, num_errs):
            A[i][col] = gf_mul(A[i][col], inv)
        B_vec[i] = gf_mul(B_vec[i], inv)
        for row in range(num_errs):
            if row != i and A[row][i] != 0:
                factor = A[row][i]
                for col in range(i, num_errs):
                    A[row][col] ^= gf_mul(A[i][col], factor)
                B_vec[row] ^= gf_mul(B_vec[i], factor)

    corrected = list(rec)
    for r, pos in enumerate(err_pos):
        corrected[pos] ^= B_vec[r]

    return bytes(corrected[:-nsym])

def interleave_bytes(byte_list, num_cols=16):
    num_rows = (len(byte_list) + num_cols - 1) // num_cols
    padded = list(byte_list) + [0] * (num_rows * num_cols - len(byte_list))
    out = []
    for c in range(num_cols):
        for r in range(num_rows):
            out.append(padded[r * num_cols + c])
    return bytes(out)

def deinterleave_bytes(byte_list, orig_len, num_cols=16):
    num_rows = len(byte_list) // num_cols
    grid = [[0] * num_cols for _ in range(num_rows)]
    idx = 0
    for c in range(num_cols):
        for r in range(num_rows):
            grid[r][c] = byte_list[idx]
            idx += 1
    flat = []
    for r in range(num_rows):
        for c in range(num_cols):
            flat.append(grid[r][c])
    return bytes(flat[:orig_len])
