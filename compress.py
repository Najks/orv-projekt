import numpy as np
from PIL import Image
import math
import struct
from collections import deque

delta_count = 0

def write_bits(value, num_bits, output_bytes, bit_position):
    current_byte = bit_position[0]
    num_bits_filled = bit_position[1]

    while num_bits > 0:
        # Izračun koliko bitov še lahko zapišemo v trenutni byte
        remaining_bits = 8 - num_bits_filled
        # Določimo koliko bitov bomo zapisali v tem koraku
        bits_to_write = min(num_bits, remaining_bits)
        current_byte = (current_byte << bits_to_write) | ((value >> (num_bits - bits_to_write)) & ((1 << bits_to_write) - 1))
        num_bits_filled += bits_to_write
        num_bits -= bits_to_write

        if num_bits_filled == 8:
            output_bytes.append(current_byte)
            current_byte = 0
            num_bits_filled = 0

    bit_position[0] = current_byte
    bit_position[1] = num_bits_filled

def flush_bits(output_bytes, bit_position):
    current_byte = bit_position[0]
    num_bits_filled = bit_position[1]

    if num_bits_filled > 0:
        output_bytes.append(current_byte << (8 - num_bits_filled))
    bit_position[0] = 0
    bit_position[1] = 0

def Predict(P, X, Y):
    E = np.zeros(X * Y, dtype=int)
    for y in range(Y):
        for x in range(X):
            i = y * X + x
            curr_val = int(P[y, x])
            if x == 0 and y == 0:  # Zgornji levi piksel
                E[i] = curr_val
            elif y == 0:  # Prva vrstica
                E[i] = int(P[y, x - 1]) - curr_val
            elif x == 0:  # Prvi stolpec
                E[i] = int(P[y - 1, x]) - curr_val
            else:  # Splošni primer
                left = int(P[y, x - 1])  # Piksel levo
                above = int(P[y - 1, x])  # Piksel zgoraj
                above_left = int(P[y - 1, x - 1])  # Piksel zgoraj levo
                # Izvedba prediktivnega algoritma
                if above_left >= max(left, above):
                    predicted = min(left, above)
                elif above_left <= min(left, above):
                    predicted = max(left, above)
                else:
                    predicted = left + above - above_left
                # Izračun napake napovedi
                E[i] = predicted - curr_val
    return E

def Encode(output_bytes, bit_position, delta, range_val):
    # Izračunaj število bitov, potrebnih za zapis delta
    num_bits = math.ceil(math.log2(range_val)) if range_val > 0 else 1
    # Pisanje delta vrednosti v output_bytes
    write_bits(delta, num_bits, output_bytes, bit_position)
    # Povečanje globalnega štetja delte
    global delta_count
    delta_count += 1

def InterpolativeCoding(output_bytes, bit_position, C, L, H):
    # Inicializacija vrste z začetnim intervalom
    queue = deque([(L, H)])
    while queue:
        current_L, current_H = queue.popleft()
        # Če je interval večji od 1, nadaljujemo z kodiranjem
        if current_H - current_L > 1:
            m = (current_L + current_H) // 2  # Sredina intervala
            delta = C[m] - C[current_L]
            range_val = C[current_H] - C[current_L] + 1
            # Klic funkcije Encode za zapis delte
            Encode(output_bytes, bit_position, delta, range_val)
            queue.extend([(current_L, m), (m, current_H)])

def Compress(P, X, Y):
    # Inicializacija globalne spremenljivke za štetje delte
    global delta_count
    delta_count = 0

    # Izračun napak napovedi za vse piksle
    E = Predict(P, X, Y)

    # Preslikava napak napovedi v nenegativna cela števila
    # Pozitivne napake so pomnožene z 2, negativne z 2 in zmanjšane za 1
    N = np.where(E >= 0, 2 * E, 2 * np.abs(E) - 1)

    # Izračun kumulativne vsote napak
    C = np.cumsum(N)

    # Inicializacija izhodnih podatkov in pozicije za pisanje bitov
    output_bytes = bytearray()
    bit_position = [0, 0]  # [trenutni_byte, bit_offset]

    # Zapis glave: >HHII
    header = struct.pack('>HHII', X, Y, C[0], C[-1])
    output_bytes.extend(header)

    # Zapis števila elementov v kumulativnem seštevku (32 bitov)
    write_bits(len(C), 32, output_bytes, bit_position)

    # Izvajanje interpolacijskega kodiranja na kumulativnem seštevku
    InterpolativeCoding(output_bytes, bit_position, C, 0, len(C) - 1)

    # Izpraznitev preostalih bitov v output_bytes
    flush_bits(output_bytes, bit_position)

    # Vračanje komprimirane podatke kot bytes objekt
    return bytes(output_bytes)

def read_grayscale_bmp(image_path):
    # Odprite BMP sliko in jo pretvorite v grayscale
    with Image.open(image_path) as img:
        img = img.convert('L')  # Pretvorba v grayscale
        P = np.array(img)  # Pretvorba v NumPy array
        return P, P.shape[1], P.shape[0]  # (array, širina, višina)

def write_compressed_file(compressed_data, output_path):
    # Zapišite komprimirane podatke v datoteko
    with open(output_path, 'wb') as f:
        f.write(compressed_data)
