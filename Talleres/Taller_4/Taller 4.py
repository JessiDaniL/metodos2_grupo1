import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random

"Punto 1"

# 1.a

def g_x(x,n=10,alpha=4/5):
    return sum(np.exp(-(x - k)**2 * k) / k**alpha for k in range(1, n+1))

def metropolis_hastings(g, num_samples=1000000, proposal_width=1.0, x_init=5):
    samples = []
    x = x_init
    for _ in range(num_samples):
        x_proposed = x + np.random.uniform(-proposal_width, proposal_width)
        acceptance_ratio = g(x_proposed) / g(x) if g(x) > 0 else 0
        if np.random.rand() < acceptance_ratio:
            x = x_proposed
        samples.append(x)
    return np.array(samples)

samples = metropolis_hastings(g_x)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=200, density=True, color='b', edgecolor='black')
plt.xlabel("x")
plt.title("Histograma de muestras generadas")
plt.savefig('Talleres\Taller_4\ 1.a.pdf')

# 1.b

pi_sqrt= np.sqrt(np.pi)

def gaussiana(x):
    return np.exp(-x**2)

div= gaussiana(samples)/g_x(samples)
suma = np.sum(div)

# Calcular A y su incertidumbre
mean_s = np.mean(div)
std_s = np.std(div, ddof=1)
A = np.sqrt(np.pi) / mean_s
sigma_A = np.sqrt(np.pi) * std_s / (np.sqrt(1000000) * mean_s)

print(f"1.b) {sigma_A:.6f}")

"Punto 2"

D1= 50
D2 = 50
lam = 670e-7
A = 0.04
a = 0.01
d = 0.1

N = 100000

#Intereferencia de Fresnel

x_muestras = np.random.uniform(-A/2, A/2, N)
y_muestras = np.concatenate([
    np.random.uniform(-d/2 - a/2, -d/2 + a/2, N // 2),
    np.random.uniform(d/2 - a/2, d/2 + a/2, N // 2)
])

z_muestras = np.linspace(-0.4, 0.4, 500)

def integral_Fresnel(x_muestras, y_muestras, z, D1, D2, lam):
    parte_1 = (2 * np.pi / lam) * (D1 + D2)
    parte_xy = (np.pi / (lam * D1)) * (x_muestras - y_muestras) ** 2
    parte_zy = (np.pi / (lam * D2)) * (z - y_muestras) ** 2 

    integral = np.sum(np.exp(1j * (parte_1 + parte_xy + parte_zy))) / N
    return np.abs(integral) ** 2

intensidad_f = np.array([integral_Fresnel(x_muestras, y_muestras, z, D1, D2, lam) for z in z_muestras])
intensidad_normalizada_f = intensidad_f/np.max(intensidad_f)

#Modo Clásico

angulo = np.arctan(z_muestras/D2)

intensidad_c = (np.cos(((np.pi * d)/lam)*np.sin(angulo)))**2 * (np.sinc((a/lam)*np.sin(angulo)))**2

intensidad_normalizada_c = intensidad_c/np.max(intensidad_c)

plt.figure(figsize=(8, 5))
plt.plot(z_muestras, intensidad_normalizada_f, label='Interferencia de Fresnel', color='b')
plt.plot(z_muestras, intensidad_normalizada_c, label='Intensidad clásica', linestyle='--', color='r')
plt.xlabel("Posición en la pantalla (cm)")
plt.ylabel("Intensidad normalizada")
plt.legend()
plt.title("Comparación de la interferecia de Fresnel y la intensidad clásica")
plt.savefig('Talleres\Taller_4\ 2.pdf')

"Punto 3"

"Punto 4"

# 4.a

# Cargar el texto
with open('Talleres\Taller_4\metamorfosis.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Limpieza del texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Mantener solo letras y espacios
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios múltiples
    return text

cleaned_text = clean_text(text)

# Guardar el texto limpio
with open('Talleres\Taller_4\metamorfosis_clean.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

# 4.b

# Parámetro n-grama
n = 3  # Cambia este valor para probar diferentes modelos

# Construcción de la tabla de frecuencias
def build_frequency_table(text, n):
    freq_table = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - n):
        prefix = text[i:i+n]
        next_char = text[i+n]
        freq_table[prefix][next_char] += 1
    return freq_table

freq_table = build_frequency_table(text, n)

# Normalización de la tabla para obtener probabilidades
def normalize_table(freq_table):
    prob_table = {}
    for prefix, next_chars in freq_table.items():
        total = sum(next_chars.values())
        prob_table[prefix] = {char: count/total for char, count in next_chars.items()}
    return prob_table

prob_table = normalize_table(freq_table)

# Generación de texto
def generate_text(prob_table, n, m=1500):
    import random
    start = random.choice(list(prob_table.keys()))
    output = start
    
    for _ in range(m - n):
        if start not in prob_table:
            break
        next_char = random.choices(list(prob_table[start].keys()), weights=prob_table[start].values())[0]
        output += next_char
        start = output[-n:]
    
    return output

# Generar un texto de 1500 caracteres
generated_text = generate_text(prob_table, n)

# 4.c 

# Cargar el texto
with open('Talleres\Taller_4\metamorfosis_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Parámetro n-grama
n_values = range(1, 8)  # Probar valores de n desde 1 hasta 7
word_list = set()

# Cargar lista de palabras en inglés
with open('Talleres\Taller_4\words_alpha.txt', 'r', encoding='utf-8') as f:
    word_list = set(f.read().splitlines())

# Construcción de la tabla de frecuencias
def build_frequency_table(text, n):
    freq_table = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - n):
        prefix = text[i:i+n]
        next_char = text[i+n]
        freq_table[prefix][next_char] += 1
    return freq_table

# Normalización de la tabla para obtener probabilidades
def normalize_table(freq_table):
    prob_table = {}
    for prefix, next_chars in freq_table.items():
        total = sum(next_chars.values())
        prob_table[prefix] = {char: count/total for char, count in next_chars.items()}
    return prob_table

# Generación de texto
def generate_text(prob_table, n, m=1500):
    start = random.choice(list(prob_table.keys()))
    output = start
    
    for _ in range(m - n):
        if start not in prob_table:
            break
        next_char = random.choices(list(prob_table[start].keys()), weights=prob_table[start].values())[0]
        output += next_char
        start = output[-n:]
    
    return output

# Evaluación del porcentaje de palabras válidas
def evaluate_generated_text(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    valid_words = [word for word in words if word in word_list]
    return len(valid_words) / len(words) * 100 if words else 0

results = {}
for n in n_values:
    freq_table = build_frequency_table(text, n)
    prob_table = normalize_table(freq_table)
    generated_text = generate_text(prob_table, n)
    valid_percentage = evaluate_generated_text(generated_text)
    results[n] = valid_percentage
    
    # Guardar el texto generado
    with open(r'Talleres\Taller_4\textos\gen_text_n{n}.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)

# Graficar resultados
plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
plt.xlabel('n-grama')
plt.ylabel('Porcentaje de palabras válidas')
plt.title('Evaluación de generación de texto con cadenas de Markov')
plt.grid()
plt.savefig('Talleres\Taller_4\ 4.pdf')

print("4.c) El número de n-gramas adecuado para generar texto con cadenas de Markov es a partir de 4, donde tiene un porcentaje de palabras válidas de {:.2f}%".format(results[4]))
