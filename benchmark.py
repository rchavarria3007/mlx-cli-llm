import sys
from mlx_lm import load, generate

# Definição do modelo otimizado pela comunidade MLX
model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"

print("🔄 Carregando o modelo na Memória Unificada do M5... (Aguarde)")
model, tokenizer = load(model_path)
print("✅ Modelo carregado e pronto!\n")

# Histórico da conversa usando a estrutura de chat do modelo
messages = [
    {
        "role": "system", 
        "content": "Você é um assistente de IA focado em engenharia de dados. Responda em português, de forma direta e técnica."
    }
]

print("🤖 Chat local iniciado. Digite 'sair' para encerrar.\n")


while True:
    user_input = input("Você 🧑‍💻: ")
    if user_input.lower() in ["sair", "exit"]:
        print("Encerrando chat local...")
        break
        
    if not user_input.strip():
        continue
        
    messages.append({"role": "user", "content": user_input})
    
    # Aplica o template de chat (ChatML) para o modelo entender o contexto
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("Bot MLX 🤖: ", end="", flush=True)
    
    # Gerando a resposta com streaming (exibindo token por token no terminal do IDE)
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        verbose=True   # Oculta logs internos de performance durante o chat
    )
    
    print(response)
    print("\n" + "—" * 50 + "\n")
    
    # Salva a resposta do assistente no histórico para manter o contexto da conversa
    messages.append({"role": "assistant", "content": response})