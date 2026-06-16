#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"


@dataclass
class Config:
    MODEL_NAME: str = "klue/roberta-small"
    DATA_DIR: str = "./learning_data"
    TEST_NORMAL_CSV: str = "./test_data/normal.csv"
    TEST_TOXIC_CSV: str = "./test_data/toxic.csv"
    NEW_WORDS_FILE: str = "./new_words.txt"
    OUTPUT_DIR: str = "./outputs_low_spec"

    USE_TAPT: bool = True
    USE_NEW_WORDS: bool = True
    FORCE_RETRAIN_TAPT: bool = False

    MAX_LEN: int = 64
    MAX_CHARS: int = 64

    BATCH_SIZE: int = 16
    GRADIENT_ACCUMULATION_STEPS: int = 2
    EPOCHS: int = 4
    VALID_RATIO: float = 0.05
    EARLY_STOPPING_PATIENCE: int = 2
    NUM_WORKERS: int = 0

    LR_BODY: float = 2e-5
    LR_HEAD: float = 5e-5
    LLRD_DECAY: float = 0.90
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.10
    MAX_GRAD_NORM: float = 1.0

    TAPT_EPOCHS: int = 2
    TAPT_BATCH_SIZE: int = 16
    TAPT_GRADIENT_ACCUMULATION_STEPS: int = 2
    TAPT_LR: float = 2e-5
    TAPT_MLM_PROBABILITY: float = 0.15

    SEED: int = 42

    @property
    def tapt_dir(self) -> str:
        return str(Path(self.OUTPUT_DIR) / "tapt_model")

    @property
    def final_model_dir(self) -> str:
        return str(Path(self.OUTPUT_DIR) / "final_model")

    @property
    def best_state_path(self) -> str:
        return str(Path(self.OUTPUT_DIR) / "best_model_state.pt")

    @property
    def test_error_path(self) -> str:
        return str(Path(self.OUTPUT_DIR) / "test_wrong_predictions.csv")


cfg = Config()

LABEL2ID = {"normal": 0, "toxic": 1}
ID2LABEL = {0: "normal", 1: "toxic"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if IS_CUDA:
        torch.cuda.manual_seed_all(seed)


def preprocess_text(text: object) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u200B-\u200D\u2060\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_json_strings(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON 읽기 실패: {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"JSON 최상위 형식은 문자열 배열이어야 합니다: {path}")
    return [x for x in data if isinstance(x, str)]


def read_text_csv(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"테스트 CSV가 없습니다: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"CSV에 'text' 열이 필요합니다: {path}")
        return [row["text"] for row in reader if row.get("text") is not None]


def clean_and_filter(
    texts: Iterable[str],
    max_chars: int,
    source_name: str,
) -> List[str]:
    kept: List[str] = []
    seen = set()
    empty_count = 0
    too_long_count = 0
    duplicate_count = 0

    for raw in texts:
        text = preprocess_text(raw)
        if not text:
            empty_count += 1
            continue
        if len(text) > max_chars:
            too_long_count += 1
            continue
        if text in seen:
            duplicate_count += 1
            continue
        seen.add(text)
        kept.append(text)

    print(
        f"  {source_name}: 유지 {len(kept):,} | "
        f"빈값 {empty_count:,} | {max_chars}자 초과 {too_long_count:,} | "
        f"중복 {duplicate_count:,}"
    )
    return kept


def hash_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in sorted(set(texts)):
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_train_val_split(
    texts: List[str], labels: List[int], valid_ratio: float, seed: int
) -> Tuple[List[str], List[str], List[int], List[int]]:
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError(f"학습 데이터에 두 라벨이 모두 필요합니다: {dict(counts)}")

    stratify = labels if min(counts.values()) >= 2 else None
    return train_test_split(
        texts,
        labels,
        test_size=valid_ratio,
        random_state=seed,
        stratify=stratify,
    )


def load_and_prepare_data() -> Dict[str, List]:
    data_dir = Path(cfg.DATA_DIR)

    print("\n[테스트 데이터 로드]")
    test_normal = clean_and_filter(
        read_text_csv(Path(cfg.TEST_NORMAL_CSV)), cfg.MAX_CHARS, "test normal"
    )
    test_toxic = clean_and_filter(
        read_text_csv(Path(cfg.TEST_TOXIC_CSV)), cfg.MAX_CHARS, "test toxic"
    )

    test_toxic_set = set(test_toxic)
    test_overlap = set(test_normal) & test_toxic_set
    if test_overlap:
        print(f"  경고: 테스트 라벨 충돌 {len(test_overlap):,}건 -> toxic 우선")
        test_normal = [x for x in test_normal if x not in test_overlap]

    test_texts = test_normal + test_toxic
    test_labels = [LABEL2ID["normal"]] * len(test_normal) + [
        LABEL2ID["toxic"]
    ] * len(test_toxic)
    test_text_set = set(test_texts)

    print("\n[학습 원본 데이터 로드]")
    train_normal = clean_and_filter(
        read_json_strings(data_dir / "normal.json"), cfg.MAX_CHARS, "train normal"
    )
    train_toxic = clean_and_filter(
        read_json_strings(data_dir / "toxic.json"),
        cfg.MAX_CHARS,
        "train toxic",
    )
    train_toxic_set = set(train_toxic)

    label_overlap = set(train_normal) & train_toxic_set
    if label_overlap:
        print(f"  경고: 학습 라벨 충돌 {len(label_overlap):,}건 -> toxic 우선")
        train_normal = [x for x in train_normal if x not in label_overlap]

    before_normal = len(train_normal)
    before_toxic = len(train_toxic)
    train_normal = [x for x in train_normal if x not in test_text_set]
    train_toxic = [x for x in train_toxic if x not in test_text_set]
    print(
        "\n[테스트 데이터 누수 제거]\n"
        f"  normal 제거: {before_normal - len(train_normal):,}건\n"
        f"  toxic 제거: {before_toxic - len(train_toxic):,}건"
    )

    all_train_texts = train_normal + train_toxic
    all_train_labels = [LABEL2ID["normal"]] * len(train_normal) + [
        LABEL2ID["toxic"]
    ] * len(train_toxic)

    train_texts, val_texts, train_labels, val_labels = make_train_val_split(
        all_train_texts, all_train_labels, cfg.VALID_RATIO, cfg.SEED
    )

    chat_log = clean_and_filter(
        read_json_strings(data_dir / "chat_log.json"), cfg.MAX_CHARS, "TAPT chat_log"
    )
    tapt_exclude = test_text_set | set(val_texts)
    tapt_texts = list(
        dict.fromkeys(train_texts + [x for x in chat_log if x not in tapt_exclude])
    )

    print("\n[최종 데이터 수]")
    print(
        f"  train: {len(train_texts):,} "
        f"(normal={train_labels.count(0):,}, toxic={train_labels.count(1):,})"
    )
    print(
        f"  validation: {len(val_texts):,} "
        f"(normal={val_labels.count(0):,}, toxic={val_labels.count(1):,})"
    )
    print(
        f"  test: {len(test_texts):,} "
        f"(normal={test_labels.count(0):,}, toxic={test_labels.count(1):,})"
    )
    print(f"  TAPT corpus: {len(tapt_texts):,}")

    if not train_texts or not val_texts or not test_texts:
        raise ValueError("train/validation/test 중 빈 데이터셋이 있습니다.")

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "tapt_texts": tapt_texts,
    }


def load_new_words(path: Path) -> List[str]:
    if not path.exists():
        print(f"  신규 단어집 없음: {path}")
        return []

    words: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            word = preprocess_text(raw)
            if word and len(word) <= cfg.MAX_CHARS and word not in seen:
                seen.add(word)
                words.append(word)
    return words


def build_tokenizer() -> Tuple[object, int]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    added_count = 0

    if cfg.USE_NEW_WORDS:
        words = load_new_words(Path(cfg.NEW_WORDS_FILE))
        added_count = tokenizer.add_tokens(words)
        print(
            f"\n[신규 단어집] 입력 {len(words):,}개, 실제 추가 {added_count:,}개, "
            f"vocab={len(tokenizer):,}"
        )
    else:
        print("\n[신규 단어집] OFF")

    return tokenizer, added_count


class MLMDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_len: int):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.tokenizer(
            self.texts[index],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )


def tapt_fingerprint(texts: Sequence[str], tokenizer) -> Dict[str, object]:
    return {
        "model_name": cfg.MODEL_NAME,
        "max_len": cfg.MAX_LEN,
        "epochs": cfg.TAPT_EPOCHS,
        "mlm_probability": cfg.TAPT_MLM_PROBABILITY,
        "vocab_size": len(tokenizer),
        "new_words_hash": file_sha256(Path(cfg.NEW_WORDS_FILE))
        if cfg.USE_NEW_WORDS
        else "disabled",
        "corpus_hash": hash_texts(texts),
    }


def cached_tapt_is_valid(tapt_dir: Path, fingerprint: Dict[str, object]) -> bool:
    metadata_path = tapt_dir / "tapt_metadata.json"
    model_exists = (tapt_dir / "model.safetensors").exists() or (
        tapt_dir / "pytorch_model.bin"
    ).exists()
    if not metadata_path.exists() or not model_exists:
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
        return old.get("fingerprint") == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def perform_tapt(tokenizer, texts: Sequence[str]) -> str:
    if not cfg.USE_TAPT:
        print("\n[TAPT] OFF")
        return ""
    if not texts:
        raise ValueError("TAPT에 사용할 문장이 없습니다.")

    tapt_dir = Path(cfg.tapt_dir)
    fingerprint = tapt_fingerprint(texts, tokenizer)

    if not cfg.FORCE_RETRAIN_TAPT and cached_tapt_is_valid(tapt_dir, fingerprint):
        print(f"\n[TAPT] 기존 모델 재사용: {tapt_dir}")
        return str(tapt_dir)

    print("\n[TAPT] 표준 MLM 학습 시작")
    print(
        f"  corpus={len(texts):,}, epochs={cfg.TAPT_EPOCHS}, "
        f"batch={cfg.TAPT_BATCH_SIZE}, max_len={cfg.MAX_LEN}"
    )

    if tapt_dir.exists():
        shutil.rmtree(tapt_dir)
    tapt_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMaskedLM.from_pretrained(cfg.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    dataset = MLMDataset(texts, tokenizer, cfg.MAX_LEN)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.TAPT_MLM_PROBABILITY,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.TAPT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=IS_CUDA,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.TAPT_LR, weight_decay=cfg.WEIGHT_DECAY)
    updates_per_epoch = max(
        1,
        (len(loader) + cfg.TAPT_GRADIENT_ACCUMULATION_STEPS - 1)
        // cfg.TAPT_GRADIENT_ACCUMULATION_STEPS,
    )
    total_updates = updates_per_epoch * cfg.TAPT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * cfg.WARMUP_RATIO),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=IS_CUDA)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg.TAPT_EPOCHS):
        model.train()
        running_loss = 0.0
        valid_steps = 0
        started = time.time()

        for step, batch in enumerate(loader):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                output = model(**batch, return_dict=True)
                loss = (output.loss if hasattr(output, "loss") else output[0]) / cfg.TAPT_GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            should_update = (
                (step + 1) % cfg.TAPT_GRADIENT_ACCUMULATION_STEPS == 0
                or (step + 1) == len(loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * cfg.TAPT_GRADIENT_ACCUMULATION_STEPS
            valid_steps += 1

        print(
            f"  epoch {epoch + 1}/{cfg.TAPT_EPOCHS} | "
            f"loss={running_loss / max(1, valid_steps):.4f} | "
            f"time={time.time() - started:.1f}s"
        )

    model.save_pretrained(tapt_dir)
    tokenizer.save_pretrained(tapt_dir)
    with (tapt_dir / "tapt_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint}, f, ensure_ascii=False, indent=2)

    del model, loader, dataset, optimizer, scheduler, scaler
    if IS_CUDA:
        torch.cuda.empty_cache()

    print(f"  저장: {tapt_dir}")
    return str(tapt_dir)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_len: int,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "text": self.texts[index],
        }


def create_loader(
    texts: Sequence[str],
    labels: Sequence[int],
    tokenizer,
    shuffle: bool,
) -> DataLoader:
    dataset = ClassificationDataset(texts, labels, tokenizer, cfg.MAX_LEN)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=IS_CUDA,
    )


def evaluate(model, loader: DataLoader, loss_fn: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[List[float]] = []
    texts_all: List[str] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = loss_fn(logits, labels)

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            total_loss += loss.item()
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
            texts_all.extend(batch["text"])

    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision_macro": precision_score(
            labels_all, preds_all, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels_all, preds_all, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(labels_all, preds_all, average="macro", zero_division=0),
        "f1_toxic": f1_score(labels_all, preds_all, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(labels_all, preds_all, labels=[0, 1]),
        "labels": labels_all,
        "preds": preds_all,
        "probs": probs_all,
        "texts": texts_all,
    }


def print_metrics(name: str, result: Dict[str, object]) -> None:
    print(
        f"  {name} | loss={result['loss']:.4f} | "
        f"acc={result['accuracy']:.4f} | "
        f"macro_f1={result['f1_macro']:.4f} | "
        f"toxic_f1={result['f1_toxic']:.4f}"
    )
    print(f"  confusion matrix [normal, toxic]:\n{result['confusion_matrix']}")


def save_wrong_predictions(result: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["text", "true_label", "pred_label", "normal_prob", "toxic_prob"]
        )
        for text, true_id, pred_id, probs in zip(
            result["texts"], result["labels"], result["preds"], result["probs"]
        ):
            if true_id != pred_id:
                writer.writerow(
                    [
                        text,
                        ID2LABEL[true_id],
                        ID2LABEL[pred_id],
                        f"{probs[0]:.6f}",
                        f"{probs[1]:.6f}",
                    ]
                )


def _split_decay_params(
    named_params: Iterable[Tuple[str, nn.Parameter]],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_llrd_groups(model) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    used_param_ids = set()

    def add_group(named_params, lr: float) -> None:
        named_params = list(named_params)
        decay, no_decay = _split_decay_params(named_params)
        if decay:
            groups.append(
                {"params": decay, "lr": lr, "weight_decay": cfg.WEIGHT_DECAY}
            )
            used_param_ids.update(id(p) for p in decay)
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
            used_param_ids.update(id(p) for p in no_decay)

    add_group(model.classifier.named_parameters(prefix="classifier"), cfg.LR_HEAD)

    layers = model.roberta.encoder.layer
    num_layers = len(layers)
    for layer_index in range(num_layers - 1, -1, -1):
        depth_from_top = num_layers - 1 - layer_index
        layer_lr = cfg.LR_BODY * (cfg.LLRD_DECAY ** depth_from_top)
        add_group(
            layers[layer_index].named_parameters(prefix=f"roberta.encoder.layer.{layer_index}"),
            layer_lr,
        )

    embedding_lr = cfg.LR_BODY * (cfg.LLRD_DECAY ** num_layers)
    add_group(
        model.roberta.embeddings.named_parameters(prefix="roberta.embeddings"),
        embedding_lr,
    )

    remaining = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in used_param_ids
    ]
    if remaining:
        add_group(remaining, cfg.LR_BODY)

    return groups


def train_classifier(tokenizer, model_source: str, data: Dict[str, List]) -> None:
    print("\n[분류 모델 로드]")
    print(f"  source: {model_source}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    train_loader = create_loader(
        data["train_texts"], data["train_labels"], tokenizer, shuffle=True
    )
    val_loader = create_loader(
        data["val_texts"], data["val_labels"], tokenizer, shuffle=False
    )
    test_loader = create_loader(
        data["test_texts"], data["test_labels"], tokenizer, shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(build_llrd_groups(model))

    updates_per_epoch = max(
        1,
        (len(train_loader) + cfg.GRADIENT_ACCUMULATION_STEPS - 1)
        // cfg.GRADIENT_ACCUMULATION_STEPS,
    )
    total_updates = updates_per_epoch * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * cfg.WARMUP_RATIO),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=IS_CUDA)

    best_f1 = -1.0
    patience = 0
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n[분류 학습 시작]")
    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        valid_steps = 0
        started = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = loss_fn(logits, labels)
                scaled_loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            should_update = (
                (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0
                or (step + 1) == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            valid_steps += 1

        val_result = evaluate(model, val_loader, loss_fn)
        print(
            f"\n  epoch {epoch + 1}/{cfg.EPOCHS} | "
            f"train_loss={running_loss / max(1, valid_steps):.4f} | "
            f"time={time.time() - started:.1f}s"
        )
        print_metrics("validation", val_result)

        if val_result["f1_macro"] > best_f1:
            best_f1 = float(val_result["f1_macro"])
            patience = 0
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, cfg.best_state_path)
            print(f"  best 저장: macro_f1={best_f1:.4f}")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOPPING_PATIENCE:
                print("  early stopping")
                break

    if not Path(cfg.best_state_path).exists():
        raise RuntimeError("최적 모델 상태가 저장되지 않았습니다.")

    try:
        best_state = torch.load(
            cfg.best_state_path, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        best_state = torch.load(cfg.best_state_path, map_location=DEVICE)
    model.load_state_dict(best_state)

    print("\n[최종 테스트]")
    test_result = evaluate(model, test_loader, loss_fn)
    print_metrics("test", test_result)
    save_wrong_predictions(test_result, Path(cfg.test_error_path))

    final_dir = Path(cfg.final_model_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    run_metadata = {
        "config": asdict(cfg),
        "labels": LABEL2ID,
        "best_validation_macro_f1": best_f1,
        "test_metrics": {
            "accuracy": test_result["accuracy"],
            "precision_macro": test_result["precision_macro"],
            "recall_macro": test_result["recall_macro"],
            "f1_macro": test_result["f1_macro"],
            "f1_toxic": test_result["f1_toxic"],
            "confusion_matrix": test_result["confusion_matrix"].tolist(),
        },
    }
    with (final_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    print("\n[완료]")
    print(f"  최종 모델: {final_dir}")
    print(f"  테스트 오분류: {cfg.test_error_path}")


def parse_on_off(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"on", "true", "1", "yes"}:
        return True
    if lowered in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("on 또는 off를 입력하세요.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="한국어 악성 채팅 2 라벨 분류 학습")
    parser.add_argument(
        "--tapt",
        type=parse_on_off,
        default=cfg.USE_TAPT,
        metavar="on|off",
        help="TAPT 사용 여부",
    )
    parser.add_argument(
        "--new-words",
        type=parse_on_off,
        default=cfg.USE_NEW_WORDS,
        metavar="on|off",
        help="new_words.txt 토큰 추가 여부",
    )
    parser.add_argument(
        "--force-retrain-tapt",
        action="store_true",
        help="캐시가 있어도 TAPT를 다시 학습",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.USE_TAPT = args.tapt
    cfg.USE_NEW_WORDS = args.new_words
    cfg.FORCE_RETRAIN_TAPT = args.force_retrain_tapt

    print("=" * 72)
    print("한국어 악성 채팅 2분류 학습")
    print("klue/roberta-small + optional TAPT/new_words + LLRD")
    print("=" * 72)
    print(f"device={DEVICE}, TAPT={cfg.USE_TAPT}, new_words={cfg.USE_NEW_WORDS}")
    print(f"MAX_CHARS={cfg.MAX_CHARS}, MAX_LEN={cfg.MAX_LEN}")

    set_seed(cfg.SEED)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    data = load_and_prepare_data()
    tokenizer, _ = build_tokenizer()
    tapt_dir = perform_tapt(tokenizer, data["tapt_texts"])
    model_source = tapt_dir if tapt_dir else cfg.MODEL_NAME
    train_classifier(tokenizer, model_source, data)


if __name__ == "__main__":
    main()
