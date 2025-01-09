import numpy as np
from PIL import Image
import math
import sys
import struct
from collections import deque

sys.setrecursionlimit(10000)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0  # Od MSB proti LSB

    def read_bits(self, num_bits):
        """Prebere določeno število bitov."""
        value = 0
        while num_bits > 0:
            if self.byte_pos >= len(self.data):
                raise ValueError("Poskus branja presega konec podatkov.")
            current_byte = self.data[self.byte_pos]
            remaining_bits = 8 - self.bit_pos
            bits_to_read = min(num_bits, remaining_bits)
            shift = remaining_bits - bits_to_read
            bits = (current_byte >> shift) & ((1 << bits_to_read) - 1)
            value = (value << bits_to_read) | bits
            self.bit_pos += bits_to_read
            num_bits -= bits_to_read
            if self.bit_pos == 8:
                self.byte_pos += 1
                self.bit_pos = 0
        return value


def InterpolativeDecodingIterative(B, C, L, H, n):
    global decoded_deltas
    queue = deque([(L, H)])
    interval = max(n // 10, 1)
    while queue:
        current_L, current_H = queue.popleft()
        if current_H - current_L > 1:
            m = (current_L + current_H) // 2
            range_val = C[current_H] - C[current_L] + 1
            g = math.ceil(math.log2(range_val)) if range_val > 0 else 1
            delta = B.read_bits(g)
            C[m] = C[current_L] + delta
            decoded_deltas += 1
            queue.append((current_L, m))
            queue.append((m, current_H))


def Decompress(compressed_data):
    global decoded_deltas
    decoded_deltas = 0

    # Izvleci informacije iz glave
    header_size = struct.calcsize('>HHII')
    header = compressed_data[:header_size]
    X, Y, C0, C_last = struct.unpack('>HHII', header)
    remaining_data = compressed_data[header_size:]

    B = BitReader(remaining_data)

    n = B.read_bits(32)

    # Inicializacija C
    C = [0] * n
    C[0] = C0
    C[-1] = C_last

    # Dekodiranje C
    InterpolativeDecodingIterative(B, C, 0, n - 1, n)

    # Rekonstruiraj N iz C
    N = [0] * n
    N[0] = C[0]
    for i in range(1, n):
        N[i] = C[i] - C[i - 1]

    # Obrni preslikavo iz N v E
    E = [0] * n
    for i in range(n):
        if N[i] % 2 == 0:
            E[i] = N[i] // 2
        else:
            E[i] = -(N[i] + 1) // 2

    P = np.zeros((Y, X), dtype=np.uint8)

    # Rekonstruiraj P iz E
    for y in range(Y):
        for x in range(X):
            i = y * X + x
            try:
                if x == 0 and y == 0:
                    # Zgornji levi piksel
                    predicted = 0
                    value = E[i] % 256
                    P[y, x] = value
                elif y == 0:
                    # Prva vrstica, iz levega piksla
                    predicted = int(P[y, x - 1])
                    value = (predicted - E[i]) % 256
                    P[y, x] = value
                elif x == 0:
                    # Prvi stolpec, iz zgornjega piksla
                    predicted = int(P[y - 1, x])
                    value = (predicted - E[i]) % 256
                    P[y, x] = value
                else:
                    # Splošni primer
                    p_x1_y1 = int(P[y - 1, x - 1])
                    p_x1_y = int(P[y, x - 1])
                    p_x_y1 = int(P[y - 1, x])

                    if p_x1_y1 >= max(p_x1_y, p_x_y1):
                        predicted = min(p_x1_y, p_x_y1)
                    elif p_x1_y1 <= min(p_x1_y, p_x_y1):
                        predicted = max(p_x1_y, p_x_y1)
                    else:
                        predicted = p_x1_y + p_x_y1 - p_x1_y1

                    # Pretvori napoved v celo število
                    predicted = int(predicted)
                    value = (predicted - E[i]) % 256
                    P[y, x] = value

                assert 0 <= P[y, x] <= 255, f"Vrednost piksla zunaj meja pri P[{y}, {x}] = {P[y, x]}"

            except AssertionError as ae:
                raise ae
            except Exception as e:
                raise e

    return P


def read_compressed_file(compressed_path):
    with open(compressed_path, 'rb') as f:
        data = f.read()
    return data


def write_decompressed_bmp(P, output_path):
    img = Image.fromarray(P, mode='L')
    img.save(output_path)
