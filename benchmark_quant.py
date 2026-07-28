import time
import mlx.core as mx
from mlx_lm import load, generate

# 1. Caminho para o modelo quantizado localmente
model_path = "./models/qwen-1.5b-4bit"

print("📦 Carregando o modelo quantizado em 4-bit na RAM/GPU...")
start_load = time.perf_counter()

model, tokenizer = load(model_path)

# Sincroniza a GPU para medir o tempo exato de carregamento dos pesos
mx.eval(model.parameters())
load_time = time.perf_counter() - start_load

print(f"✅ Modelo de 1.5B em 4-bit carregado em {load_time:.2f}s!\n")

# 2. Prompt de teste técnico
prompt = "Escreva uma função em Python para ordenar um array usando o algoritmo QuickSort, com comentários detalhados."

messages = [
    {"role": "system", "content": "Você é um assistente técnico especialista em engenharia de software e estruturas de dados."},
    {"role": "user", "content": prompt}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print("📝 Processando Prompt e Gerando Resposta:\n")
print("-" * 60)

# 3. Execução com verbose=True para capturar a taxa de tokens/segundo
start_gen = time.perf_counter()

response = generate(
    model,
    tokenizer,
    prompt=formatted_prompt,
    max_tokens=512,
    verbose=True  # Exibe o relatório de performance do MLX no final
)

total_time = time.perf_counter() - start_gen

print("-" * 60)
print(f"\n⏱️ Tempo total de execução: {total_time:.2f} segundos.")