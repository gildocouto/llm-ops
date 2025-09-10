# Hub Downloader

`hub_downloader.py` é uma ferramenta de linha de comando (CLI) para baixar **modelos** e **datasets** do [Hugging Face Hub](https://huggingface.co/), com suporte a filtros, modo offline, repositórios privados e diretórios personalizados.

---

## 📦 Instalação

```bash
pip install huggingface_hub>=0.23
```

Clone ou copie o arquivo `hub_downloader.py` para seu ambiente local.

---

## 🚀 Uso Básico

```bash
python hub_downloader.py <repo_id> [opções]
```

### Parâmetro obrigatório

* `repo_id`: identificador do repositório no Hugging Face Hub (ex.: `bert-base-uncased`, `meta-llama/Llama-3.1-8B-Instruct`).

### Principais opções

* `--local-dir`: diretório de destino (padrão: cache do HF).
* `--allow-patterns`: padrões de arquivos a incluir (pode ser usado múltiplas vezes ou em CSV).
* `--ignore-patterns`: padrões de arquivos a ignorar.
* `--revision`: branch, tag ou SHA específico.
* `--max-workers`: número de downloads em paralelo.
* `--symlinks/--no-symlinks`: ativa/desativa symlinks no destino (padrão: desativado, útil em Windows).
* `--offline/--no-offline`: modo offline (padrão: segue variável `HUGGINGFACE_HUB_OFFLINE`).
* `--log-level`: nível de log (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`).

---

## 📝 Exemplos

### 1) Download simples usando cache do HF

```bash
python hub_downloader.py bert-base-uncased
```

### 2) Salvar em diretório específico (⚠️ Windows: cuidado com aspas)

```bash
python hub_downloader.py meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "~/modelos/llama31_8b" \
  --allow-patterns "*.json,*.safetensors" --allow-patterns "tokenizer.*" \
  --ignore-patterns "*.bin" \
  --max-workers 16 --no-symlinks --log-level INFO
```

### 3) Modo offline (sem rede), fixando revisão por SHA

```bash
python hub_downloader.py bert-base-uncased \
  --revision 4b07c2f8c0b6f9ac6a6f85a0f1b9d3c0f7f2c123 \
  --offline --log-level DEBUG
```

### 4) Repositório privado (token via variável de ambiente)

No **Windows (cmd)**:

```bat
set HF_TOKEN=hf_XXXX
python hub_downloader.py org/modelo-privado --local-dir "C:/model-cache/modelo-privado"
```

No **Linux/macOS (bash/zsh)**:

```bash
export HF_TOKEN=hf_XXXX
python hub_downloader.py org/modelo-privado --local-dir "/mnt/models/modelo-privado"
```

---

## 🔧 Dicas Avançadas

* Use `--allow-patterns` para baixar apenas os arquivos realmente necessários (`*.safetensors`, `config.json`, `tokenizer.*`).
* Fixe `--revision` em um **commit SHA** para garantir reprodutibilidade.
* Configure `HF_HOME` para usar um cache compartilhado em servidores multiusuário.
* Combine `--offline` e `--revision` para rodar em ambientes **air-gapped** (sem internet).

---

## 📊 Comparação: `hub_downloader.py` x `hf_hub_download`

### ✅ `hub_downloader.py` (snapshot completo)

* Baixa **todos os arquivos relevantes** de um modelo/dataset (ou filtrados via `--allow-patterns` / `--ignore-patterns`).
* Útil para **usar o modelo localmente** sem depender da rede.
* Bom para cenários de **produção, servidores, CI/CD** e **air-gapped environments**.
* Exemplo:

  ```bash
  python hub_downloader.py bert-base-uncased --allow-patterns "*.json,*.safetensors"
  ```

### ✅ `hf_hub_download` (arquivo único)

* Baixa **apenas um arquivo específico** de um repositório.
* Ideal para quando se deseja **um config.json, tokenizer ou checkpoint único**.
* Usado diretamente dentro de scripts Python, não via CLI.
* Exemplo:

  ```python
  from huggingface_hub import hf_hub_download

  config_path = hf_hub_download(
      repo_id="bert-base-uncased",
      filename="config.json",
      revision="main"
  )
  print("Arquivo salvo em:", config_path)
  ```

👉 **Resumo:**

* Use `hub_downloader.py` quando quiser **espelhar ou materializar** o repositório completo (ou subconjunto relevante).
* Use `hf_hub_download` quando precisar apenas de **um arquivo isolado** de dentro do repositório.

---

## 📄 Licença

Uso conforme as [licenças do Hugging Face Hub](https://huggingface.co/docs/hub/repositories-licenses).
Este utilitário é fornecido como exemplo prático e pode ser adaptado conforme necessidade.

