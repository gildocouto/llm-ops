#!/usr/bin/env python3
# hub_downloader.py
# pip install huggingface_hub>=0.23
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, List

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    HfHubHTTPError,
)

logger = logging.getLogger("hub_downloader")

# ----------------------------- Utilidades CLI ----------------------------- #
def _flatten_patterns(values: Optional[List[str]]) -> Optional[List[str]]:
    """
    Aceita lista com itens possivelmente em CSV e achata em uma única lista.
    Ex.: ["*.json,*.safetensors", "tokenizer.*"] -> ["*.json", "*.safetensors", "tokenizer.*"]
    """
    if not values:
        return None
    out: List[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or None


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

# ----------------------------- Núcleo de Download ----------------------------- #
def download_model(
    repo_id: str,
    local_dir: Path | str | None = None,
    *,
    token: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    revision: str | None = None,
    max_workers: Optional[int] = None,
    offline: Optional[bool] = None,
    local_dir_use_symlinks: bool = False,
) -> Path:
    """
    Baixa um *snapshot* de um repositório (modelo/dataset) do Hugging Face.

    Parâmetros:
        repo_id: Ex.: "bert-base-uncased" ou "org/modelo".
        local_dir: Diretório de destino; se None, usa o cache do HF.
        token: Token para repositórios privados; se None, tenta HF_TOKEN/HUGGINGFACEHUB_API_TOKEN.
        allow_patterns: Padrões a incluir (["*.json", "*.safetensors", "tokenizer.*", ...]).
        ignore_patterns: Padrões a ignorar (["*.bin", ...]).
        revision: Branch/tag/commit (ex.: "main" ou SHA).
        max_workers: Paralelismo de downloads.
        offline: True para modo offline (respeita HUGGINGFACE_HUB_OFFLINE).
        local_dir_use_symlinks: True usa symlinks (bom em Linux para economizar disco).

    Retorna:
        Path do diretório final materializado.
    """
    if not repo_id or repo_id.strip() == "" or repo_id.endswith("/"):
        raise ValueError("`repo_id` inválido. Ex.: 'bert-base-uncased' ou 'org/modelo' (sem barra no final).")

    resolved_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if local_dir is not None:
        local_dir = Path(local_dir).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Iniciando download: repo_id='%s'%s%s",
        repo_id,
        f", revision='{revision}'" if revision else "",
        f", destino='{local_dir}'" if local_dir else " (cache HF)",
    )

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir) if local_dir else None,
            local_dir_use_symlinks=local_dir_use_symlinks,
            token=resolved_token,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            max_workers=max_workers,
            offline=offline,
        )
        final_path = Path(downloaded_path).resolve()
        logger.info("Download concluído em: %s", final_path)
        return final_path

    except RepositoryNotFoundError:
        logger.error("Repositório não encontrado ou sem acesso: '%s'. Verifique nome/permissões/token.", repo_id)
        raise
    except GatedRepoError:
        logger.error("Repositório com acesso controlado (gated). Aceite os termos na página do modelo/dataset.")
        raise
    except LocalEntryNotFoundError:
        logger.error("Nenhuma entrada local encontrada com os filtros. Revise allow/ignore/revision.")
        raise
    except HfHubHTTPError as e:
        logger.error("Erro HTTP ao acessar o Hub: %s", e)
        raise
    except Exception:
        logger.exception("Falha inesperada ao baixar '%s'.", repo_id)
        raise


# ----------------------------- Entrypoint CLI ----------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hub Downloader: baixa e materializa snapshots de repositórios (modelos/datasets) do Hugging Face."
    )
    # Posicionais
    parser.add_argument("repo_id", type=str, help="Ex.: 'bert-base-uncased' ou 'org/modelo'.")

    # Opcionais
    parser.add_argument("--local-dir", type=str, default=None, help="Diretório destino. Padrão: cache do HF.")
    parser.add_argument("--token", type=str, default=None, help="Token HF (privado). Padrão: HF_TOKEN/HUGGINGFACEHUB_API_TOKEN.")
    parser.add_argument("--revision", type=str, default=None, help="Branch/tag/commit. Ex.: 'main' ou SHA.")
    parser.add_argument("--max-workers", type=int, default=None, help="Paralelismo de downloads (auto se omitido).")

    # Padrões: múltiplas ocorrências ou CSV
    parser.add_argument(
        "--allow-patterns",
        action="append",
        default=None,
        help="Padrões a incluir. Use múltiplos --allow-patterns ou CSV. Ex.: --allow-patterns '*.json,*.safetensors' --allow-patterns tokenizer.*",
    )
    parser.add_argument(
        "--ignore-patterns",
        action="append",
        default=None,
        help="Padrões a ignorar. Mesma regra de --allow-patterns.",
    )

    # BooleanOptionals (requer Python 3.9+)
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except AttributeError:  # fallback p/ Python <3.9
        bool_action = "store_true"

    parser.add_argument(
        "--symlinks",
        dest="local_dir_use_symlinks",
        action=bool_action,  # --symlinks / --no-symlinks
        default=False,
        help="Usa symlinks no destino. Útil em Linux p/ economizar disco (padrão: False).",
    )
    parser.add_argument(
        "--offline",
        dest="offline",
        action=bool_action,  # --offline / --no-offline
        default=None,
        help="Modo offline (não acessa rede). Padrão: respeita HUGGINGFACE_HUB_OFFLINE.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Nível de log (padrão: INFO).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    allow = _flatten_patterns(args.allow_patterns)
    ignore = _flatten_patterns(args.ignore_patterns)

    try:
        out_path = download_model(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=args.token,
            allow_patterns=allow,
            ignore_patterns=ignore,
            revision=args.revision,
            max_workers=args.max_workers,
            offline=args.offline,
            local_dir_use_symlinks=bool(args.local_dir_use_symlinks),
        )
        print(str(out_path))  # caminho final para encadear em scripts
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
