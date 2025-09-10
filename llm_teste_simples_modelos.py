#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_textgen.py
Teste rápido de um modelo Instruct compatível com Hugging Face Transformers.
Exemplos de modelos:
- mistralai/Mistral-7B-Instruct-v0.3        (público)
- mistralai/Mixtral-8x7B-Instruct-v0.1      (público, precisa +VRAM)
- meta-llama/Llama-3.1-8B-Instruct          (requer autorização no HF)
"""

import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def infer_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        # V100 32GB lida bem com fp16
        return torch.float16
    # CPU: fp32 para simplicidade/compat
    return torch.float32

def ensure_pad_token(tokenizer):
    # Garante pad_token_id para evitar erros em generate()
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # fallback genérico
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

def build_inputs(tokenizer, device, user_msg: str):
    messages = [{"role": "user", "content": user_msg}]
    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(device)
    # Fallback: concat simples
    prompt = f"### Instrução:\n{user_msg}\n\n### Resposta:"
    return tokenizer(prompt, return_tensors="pt").to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="ID do modelo no Hugging Face Hub (compatível com transformers)."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Número máximo de tokens gerados."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de amostragem (0.0 = determinístico)."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p para nucleus sampling (usado se temperature>0)."
    )
    parser.add_argument(
        "--prompt",
        default="Quem é você? Explique suas funções.",
        help="Mensagem do usuário."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Use se o modelo exigir código remoto do repositório."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = infer_dtype(device)

    print(f"Usando modelo: {args.model_id}")
    print(f"Dispositivo: {device} | dtype: {dtype}")

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code
    )
    ensure_pad_token(tokenizer)

    print("Carregando modelo (pode levar alguns minutos)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code
    )

    # Se adicionamos pad_token dinamicamente, alinhar embeddings
    if getattr(tokenizer, "pad_token_id", None) is not None and getattr(model, "get_input_embeddings", None):
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    inputs = build_inputs(tokenizer, device, args.prompt)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.temperature and args.temperature > 0.0),
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Remove chaves None para evitar avisos
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    print("Gerando resposta...")
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decodificação: tenta recortar somente a continuação
    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[0]
    continuation = generated[input_len:] if generated.shape[0] > input_len else generated
    reply = tokenizer.decode(continuation, skip_special_tokens=True)

    print("\n--- Resposta do modelo ---")
    print(reply.strip())

if __name__ == "__main__":
    main()

# Exemplo (modelo público Mistral Instruct):
# python test_textgen.py --model-id mistralai/Mistral-7B-Instruct-v0.3 --prompt "Explique RAG em 3 passos."

# Se for usar Llama 3.1 (requer autorização no HuggingFace):
# python test_textgen.py --model-id meta-llama/Llama-3.1-8B-Instruct --prompt "Explique RAG em 3 passos."

# Geração mais criativa (amostragem ligada):
# python test_textgen.py --temperature 0.7 --top-p 0.9 --prompt "Dê um exemplo de uso de embeddings no RFB."

# Dicas rápidas
# V100 32 GB comporta bem Mistral-7B-Instruct fp16 e Llama-3.1-8B-Instruct fp16; para modelos maiores (Mixtral 8x7B) prefira bfloat16 + offload ou quantização (bitsandbytes/awq/gguf com outro runtime).
# Se o repositório exigir trust_remote_code, passe --trust-remote-code.
# Se aparecer erro de autenticação para Llama, aceite a licença no Hub e faça login (huggingface-cli login) – ou escolha um modelo público como o da Mistral.
