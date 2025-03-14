import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random

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
with open('/mnt/data/words_alpha.txt', 'r', encoding='utf-8') as f:
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
    with open(f'Talleres\Taller_4\textos\gen_text_n{n}.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)

# Graficar resultados
plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
plt.xlabel('n-grama')
plt.ylabel('Porcentaje de palabras válidas')
plt.title('Evaluación de generación de texto con cadenas de Markov')
plt.grid()
plt.savefig('Talleres\Taller_4\ 4.pdf')
plt.show()

print("Generación de textos completada. Gráfica guardada como 4.pdf")
