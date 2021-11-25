import scipy
import scipy.signal


class Const:
    EPSILON = 1e-12
    N_FFT = 512
    HOP_LENGTH = 160
    WIN_LENGTH = 512
    WINDOW = scipy.signal.hamming
    SHIFT = 0 #12
    FBIN = N_FFT // 2 + 1