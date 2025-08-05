import json
import random
import os
import time
import signal
import threading
import logging
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote


locker = False
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ModelConfig:
    vocab_size: int = 25000  # biar ap? biar bgs.
    embedding_dim: int = 384
    hidden_dim: int = 768
    num_layers: int = 4
    dropout: float = 0.15
    learning_rate: float = 0.0005
    batch_size: int = 48
    max_sequence_length: int = 256
    softmax_temperature: float = 0.8
    weight_factor: float = 1.5
    top_k: int = 50
    top_p: float = 0.85
    context_window: int = 150
    response_length_factor: float = 1.2
    min_word_freq: int = 3  # frekuansi minimal buat vocab ya jink
    fallback_responses: List[str] = None

    def __post_init__(self):
        if self.fallback_responses is None:
            self.fallback_responses = [
                "I understand what you're saying.",
                "That's an interesting point.",
                "Could you tell me more about that?",
                "I see what you mean.",
                "That makes sense.",
                "Let me think about that.",
                "I appreciate you sharing that with me.",
                "That's a good question.",
                "I'm learning from our conversation.",
                "Thank you for explaining that."
            ]

class EnhancedTextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.lemmatizer = None

        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "what's": "what is"
        }

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text(self, text: str, level: str = 'medium') -> str:
        if not text or len(text.strip()) == 0:
            return ""

        text = self.expand_contractions(text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)

        if level in ['medium', 'heavy']:
            text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\'\"]", " ", text)
            text = re.sub(r'\s+', ' ', text)
            text = text.lower().strip()

        if level == 'heavy' and self.lemmatizer:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 1]
            text = " ".join(tokens)

        return text

    def extract_features(self, text: str) -> Dict:
        tokens = word_tokenize(text.lower())
        return {
            'length': len(text),
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'punctuation_density': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        }

class ImprovedLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            config.hidden_dim * 2,
            num_heads=12,
            dropout=config.dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2)
        )

        self.output_projection = nn.Linear(config.hidden_dim * 2, config.vocab_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(param, 0, 0.1)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attention_mask)
        attended = self.layer_norm1(attended + lstm_out)
        ffn_out = self.ffn(attended)
        output = self.layer_norm2(ffn_out + attended)

        logits = self.output_projection(output)
        return logits

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(tokens, dtype=torch.long)

class EnhancedTokenizer:
    def __init__(self, vocab_size: int = 25000, min_freq: int = 3):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.word_freq = Counter()
        self.fallback_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that']

    def build_vocab(self, texts: List[str]):
        logging.info("Building enhanced vocabulary...")
        for text in tqdm(texts, desc="Counting words"):
            tokens = word_tokenize(text.lower())
            self.word_freq.update(tokens)

        filtered_words = {word: freq for word, freq in self.word_freq.items() 
                         if freq >= self.min_freq and word.isalpha()}
        most_common = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        available_slots = self.vocab_size - len(self.word2idx)
        for word, freq in most_common[:available_slots]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        for word in self.fallback_words:
            if word not in self.word2idx and len(self.word2idx) < self.vocab_size:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        logging.info(f"Built vocabulary with {len(self.word2idx)} words")
        logging.info(f"Most common words: {list(self.word2idx.keys())[3:13]}")

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        tokens = word_tokenize(text.lower())
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])

        if len(indices) > max_length - 2:
            indices = indices[:max_length-2]
        indices = [1] + indices + [2]  # <sos> + tokens + <eos>

        while len(indices) < max_length:
            indices.append(0)  # <pad>
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special and token in ['<pad>', '<sos>', '<eos>']:
                    continue
                tokens.append(token)

        return " ".join(tokens)

class MultiDatasetLoader:
    def __init__(self, max_workers: int = int(os.cpu_count()-1)):
        self.max_workers = max_workers
        self.datasets = {
            'openassistant': self._load_openassistant,
            'dialogsum': self._load_dialogsum,
            'empathetic': self._load_empathetic_dialogues,
            'persona_chat': self._load_persona_chat,
            'blended_skill': self._load_blended_skill,
            'wizard_of_wikipedia': self._load_wizard_wikipedia,
            'reddit': self._load_reddit_conversations,
            'common_crawl': self._load_common_crawl_dialogues
        }

    def load_multiple_datasets(self, dataset_names: List[str], max_samples_per_dataset: int = 10000) -> Tuple[List[str], List[Tuple[str, int]]]:
        all_texts = []
        all_sentiment_data = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self._load_single_dataset, name, max_samples_per_dataset): name 
                for name in dataset_names
            }

            for future in tqdm(as_completed(future_to_dataset), total=len(dataset_names), desc="Loading datasets"):
                dataset_name = future_to_dataset[future]
                try:
                    texts, sentiment_data = future.result()
                    all_texts.extend(texts)
                    all_sentiment_data.extend(sentiment_data)
                    logging.info(f"Loaded {len(texts)} samples from {dataset_name}")
                except Exception as e:
                    logging.error(f"Failed to load {dataset_name}: {e}")

        return all_texts, all_sentiment_data

    def _load_single_dataset(self, dataset_name: str, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        if dataset_name not in self.datasets:
            logging.warning(f"Unknown dataset: {dataset_name}")
            return [], []

        return self.datasets[dataset_name](max_samples)

    def _load_openassistant(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break

                text = row.get("text", "").strip()
                if text and len(text) > 10:
                    texts.append(text)
                    sentiment = self._analyze_sentiment(text)
                    if sentiment is not None:
                        sentiment_data.append((text, sentiment))

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading OpenAssistant: {e}")
            return [], []

    def _load_dialogsum(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("knkarthick/dialogsum", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break

                dialogue = row.get("dialogue", "").strip()
                summary = row.get("summary", "").strip()

                if dialogue:
                    texts.append(dialogue)
                    sentiment = self._analyze_sentiment(dialogue)
                    if sentiment is not None:
                        sentiment_data.append((dialogue, sentiment))

                if summary:
                    texts.append(summary)

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading DialogSum: {e}")
            return [], []

    def _load_empathetic_dialogues(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("empathetic_dialogues", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break

                utterance = row.get("utterance", "").strip()
                context = row.get("context", "").strip()

                if utterance:
                    texts.append(utterance)
                    sentiment = self._analyze_sentiment(utterance)
                    if sentiment is not None:
                        sentiment_data.append((utterance, sentiment))

                if context:
                    texts.append(context)

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading Empathetic Dialogues: {e}")
            return [], []

    def _load_persona_chat(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("bavard/personachat_truecased", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break

                history = row.get("history", [])
                candidates = row.get("candidates", [])

                for text in history + candidates:
                    if text and len(text.strip()) > 5:
                        texts.append(text.strip())
                        sentiment = self._analyze_sentiment(text)
                        if sentiment is not None:
                            sentiment_data.append((text.strip(), sentiment))

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading PersonaChat: {e}")
            return [], []

    def _load_blended_skill(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("blended_skill_talk", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break

                previous_utterance = row.get("previous_utterance", [])
                free_messages = row.get("free_messages", [])
                guided_messages = row.get("guided_messages", [])
                all_messages = previous_utterance + free_messages + guided_messages

                for msg in all_messages:
                    if isinstance(msg, str) and len(msg.strip()) > 5:
                        texts.append(msg.strip())
                        sentiment = self._analyze_sentiment(msg)
                        if sentiment is not None:
                            sentiment_data.append((msg.strip(), sentiment))

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading Blended Skill Talk: {e}")
            return [], []

    def _load_wizard_wikipedia(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("wizard_of_wikipedia", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break
                dialog = row.get("dialog", [])
                for turn in dialog:
                    text = turn.get("text", "").strip()
                    if text and len(text) > 10:
                        texts.append(text)
                        sentiment = self._analyze_sentiment(text)
                        if sentiment is not None:
                            sentiment_data.append((text, sentiment))

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading Wizard of Wikipedia: {e}")
            return [], []

    def _load_reddit_conversations(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("reddit", split="train")
            texts = []
            sentiment_data = []

            for i, row in enumerate(dataset):
                if i >= max_samples:
                    break
                content = row.get("content", "").strip()
                if content and len(content) > 30 and len(content) < 1000:
                    texts.append(content)
                    sentiment = self._analyze_sentiment(content)
                    if sentiment is not None:
                        sentiment_data.append((content, sentiment))

            return texts, sentiment_data
        except Exception as e:
            logging.error(f"Error loading Reddit conversations: {e}")
            return [], []

    def _load_common_crawl_dialogues(self, max_samples: int) -> Tuple[List[str], List[Tuple[str, int]]]:
        try:
            #nanti aja dah, cape jink
            return [], []
        except Exception as e:
            logging.error(f"Error loading Common Crawl dialogues: {e}")
            return [], []

    def _analyze_sentiment(self, text: str) -> Optional[int]:
        text_lower = text.lower()

        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 
                         'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'awesome',
                         'fantastic', 'brilliant', 'outstanding', 'superb', 'marvelous']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                         'problem', 'issue', 'error', 'wrong', 'fail', 'disappointing',
                         'frustrating', 'annoying', 'stupid', 'useless', 'worst']

        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)

        if positive_score > negative_score and positive_score > 0:
            return 1  # +
        elif negative_score > positive_score and negative_score > 0:
            return 0  # -
        else:
            return None  # g ad, goblok ini

class EnhancedNeuralChat:
    def __init__(self, config: ModelConfig = None, model_file: str = 'enhanced_ai_model.pkl'):
        self.config = config or ModelConfig()
        self.model_file = model_file
        self.preprocessor = EnhancedTextPreprocessor()
        self.tokenizer = EnhancedTokenizer(self.config.vocab_size, self.config.min_word_freq)

        # ngram, garam, garam dan madu, ahahay
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0

        # komponen sma neural, alah jmbot, ngantuk pepek
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # konteks sama personality
        self.conversation_history = []
        self.personality_bias = {}
        self.lock = threading.Lock()

        # analisis
        self.sentiment_model = None
        self.sentiment_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

        # metrik
        self.training_losses = []
        self.validation_losses = []

        self.load_model()

    def initialize_neural_model(self):
        self.model = ImprovedLSTM(self.config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        param_count = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Neural model initialized with {param_count:,} parameters")

    def train_neural_model(self, texts: List[str], epochs: int = 8, validation_split: float = 0.15):
        if not self.model:
            try:
                self.initialize_neural_model()
            except Exception as e:
                logging.error(f"Failed to initialize neural model: {e}")
                return

        if len(texts) < 50:
            logging.warning("Not enough training data for neural model. Skipping neural training.")
            return

        logging.info("Building enhanced vocabulary...")
        self.tokenizer.build_vocab(texts)

        actual_vocab_size = len(self.tokenizer.word2idx)
        if actual_vocab_size != self.config.vocab_size:
            logging.info(f"Updating vocab size from {self.config.vocab_size} to {actual_vocab_size}")
            self.config.vocab_size = actual_vocab_size
            self.initialize_neural_model()

        train_texts, val_texts = train_test_split(texts, test_size=validation_split, random_state=42)
        try:
            train_dataset = TextDataset(train_texts, self.tokenizer, self.config.max_sequence_length)
            val_dataset = TextDataset(val_texts, self.tokenizer, self.config.max_sequence_length)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        except Exception as e:
            logging.error(f"Failed to create data loaders: {e}")
            return

        self.model.train()
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            try:
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        batch = batch.to(self.device, non_blocking=True)
                        if batch.size(0) < 2 or batch.size(1) < 3:
                            continue

                        input_ids = batch[:, :-1]
                        targets = batch[:, 1:]
                        self.optimizer.zero_grad()
                        outputs = self.model(input_ids)

                        loss = F.cross_entropy(
                            outputs.reshape(-1, self.config.vocab_size),
                            targets.reshape(-1),
                            ignore_index=0,
                            label_smoothing=0.1
                        )

                        if torch.isnan(loss) or torch.isinf(loss):
                            logging.warning("Invalid loss detected, skipping batch")
                            continue

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                        })

                    except Exception as batch_e:
                        logging.warning(f"Error in batch {batch_idx}: {batch_e}")
                        continue

                if num_batches == 0:
                    logging.error("No valid batches processed")
                    break

                avg_train_loss = total_loss / num_batches
                val_loss = self.validate_neural_model(val_loader)
                self.scheduler.step()

                self.training_losses.append(avg_train_loss)
                self.validation_losses.append(val_loss)
                epoch_time = time.time() - epoch_start_time
                logging.info(
                    f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                    f'Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s'
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping after {epoch+1} epochs")
                        break

            except Exception as epoch_e:
                logging.error(f"Error in epoch {epoch+1}: {epoch_e}")
                break

        logging.info("Neural model training completed!")

    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_path = self.model_file.replace('.pkl', f'_best_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, checkpoint_path)

    def validate_neural_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(self.device, non_blocking=True)
                    if batch.size(0) < 2 or batch.size(1) < 3:
                        continue

                    input_ids = batch[:, :-1]
                    targets = batch[:, 1:]
                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(
                        outputs.reshape(-1, self.config.vocab_size),
                        targets.reshape(-1),
                        ignore_index=0
                    )

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    continue

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def generate_neural_response(self, prompt: str, max_length: int = 1000) -> str:
        if not self.model:
            return self.generate_ngram_response(prompt)

        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt, max_length=self.config.max_sequence_length//2)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            generated = tokens.copy()

            for step in range(max_length):
                if len(generated) >= self.config.max_sequence_length:
                    break

                current_input = torch.tensor(
                    [generated[-self.config.max_sequence_length:]],
                    dtype=torch.long
                ).to(self.device)

                outputs = self.model(current_input)
                logits = outputs[0, -1, :] / self.config.softmax_temperature
                next_token = self.enhanced_sample_token(logits)

                if next_token == 2:  # <eos>
                    break
                generated.append(next_token)

            response = self.tokenizer.decode(generated, skip_special=True)
            response = self.clean_response(response, prompt)

        self.model.train()
        return response

    def enhanced_sample_token(self, logits: torch.Tensor) -> int:
        logits[0] = float('-inf')
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(self.config.top_k, len(logits)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(0, top_k_indices, top_k_logits)
            logits = filtered_logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)

        if torch.sum(probs) == 0:
            valid_tokens = [i for i in range(3, len(self.tokenizer.word2idx)) 
                           if i in self.tokenizer.idx2word]
            if valid_tokens:
                return random.choice(valid_tokens)
            else:
                return 3

        next_token = torch.multinomial(probs, 1).item()
        return next_token

    def clean_response(self, response: str, prompt: str) -> str:
        if not response or len(response.strip()) == 0:
            return random.choice(self.config.fallback_responses)

        prompt_words = set(prompt.lower().split())
        response_words = response.split()

        start_idx = 0
        for i, word in enumerate(response_words):
            if word.lower() not in prompt_words:
                start_idx = i
                break

        if start_idx > 0:
            response = " ".join(response_words[start_idx:])

        response = response.strip()
        if not response:
            return random.choice(self.config.fallback_responses)

        if response and response[-1] not in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                response += '.'
        return response

    def generate_ngram_response(self, prompt: str) -> str:
        tokens = word_tokenize(prompt.lower())
        response = []
        for n in [4, 3, 2, 1]:
            if len(tokens) >= n:
                seed = tuple(tokens[-n:])
                if n == 4 and seed in self.ngram4:
                    response.extend(list(seed))
                    next_token = self.smart_ngram_sampling(self.ngram4[seed])
                    if next_token:
                        response.append(next_token)
                        break
                elif n == 3 and seed in self.trigram:
                    response.extend(list(seed))
                    next_token = self.smart_ngram_sampling(self.trigram[seed])
                    if next_token:
                        response.append(next_token)
                        break
                elif n == 2 and seed in self.trigram:
                    response.extend(list(seed))
                    next_token = self.smart_ngram_sampling(self.trigram[seed])
                    if next_token:
                        response.append(next_token)
                        break
                elif n == 1 and seed[0] in self.bigram:
                    response.append(seed[0])
                    next_token = self.smart_ngram_sampling(self.bigram[seed[0]])
                    if next_token:
                        response.append(next_token)
                        break

        if not response and self.unigram:
            common_starters = ['i', 'the', 'that', 'this', 'it', 'we', 'you']
            starter = next((word for word in common_starters if word in self.unigram), None)
            if starter:
                response.append(starter)

        desired_length = random.randint(250, 1000)
        attempts = 0
        max_attempts = desired_length * 2

        while len(response) < desired_length and attempts < max_attempts:
            attempts += 1
            next_token = None

            # 4gram lyer
            if len(response) >= 3:
                key4 = tuple(response[-3:])
                if key4 in self.ngram4:
                    next_token = self.smart_ngram_sampling(self.ngram4[key4])

            # fallbck tgram
            if not next_token and len(response) >= 2:
                key3 = tuple(response[-2:])
                if key3 in self.trigram:
                    next_token = self.smart_ngram_sampling(self.trigram[key3])

            # fallbck bgram
            if not next_token and response and response[-1] in self.bigram:
                next_token = self.smart_ngram_sampling(self.bigram[response[-1]])

            # final
            if not next_token and self.unigram:
                common_words = ['and', 'the', 'to', 'of', 'a', 'in', 'is', 'it', 'that']
                next_token = next((word for word in common_words if word in self.unigram), None)

            if next_token:
                response.append(next_token)
            else:
                break

        result = " ".join(response)
        return result if result else random.choice(self.config.fallback_responses)

    def smart_ngram_sampling(self, counter: Counter) -> Optional[str]:
        if not counter:
            return None

        biased_counter = {}
        for word, count in counter.items():
            bias = self.personality_bias.get(word, 1.0)
            biased_counter[word] = count * bias

        words = list(biased_counter.keys())
        counts = np.array(list(biased_counter.values()), dtype=float)
        if len(words) == 0:
            return None

        logits = np.log(counts + 1e-10) * self.config.weight_factor

        if self.config.top_k > 0 and len(words) > self.config.top_k:
            sorted_idx = np.argsort(logits)[::-1][:self.config.top_k]
            words = [words[i] for i in sorted_idx]
            logits = logits[sorted_idx]

        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_idx = np.where(cumulative_probs <= self.config.top_p)[0]
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[-1] + 1
            final_words = [words[sorted_idx[i]] for i in range(cutoff_idx)]
            final_probs = sorted_probs[:cutoff_idx]
            final_probs = final_probs / np.sum(final_probs)
        else:
            final_words = [words[sorted_idx[0]]]
            final_probs = np.array([1.0])

        return random.choices(final_words, weights=final_probs)[0]

    def train_ngram(self, sentence: str):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return

        with self.lock:
            # ugram
            self.unigram.update(tokens)

            # bgram
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])

            # tgram
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1

            # 4-gram
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])

            self.record_counter += 1

    def train_sentiment_model(self, texts: List[str], labels: List[int]):
        if len(texts) != len(labels) or len(texts) < 10:
            logging.warning("Insufficient data for sentiment training")
            return

        try:
            X = self.sentiment_vectorizer.fit_transform(texts)
            y = np.array(labels)

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.sentiment_model = LogisticRegression(
                max_iter=2000, 
                random_state=42,
                class_weight='balanced',
                C=0.1
            )
            self.sentiment_model.fit(X_train, y_train)
            y_pred = self.sentiment_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            logging.info(f"Sentiment model trained with accuracy: {accuracy:.4f}")

        except Exception as e:
            logging.error(f"Error training sentiment model: {e}")

    def predict_sentiment(self, text: str) -> Dict:
        if not self.sentiment_model:
            return {"sentiment": "unknown", "confidence": 0.0}

        try:
            X = self.sentiment_vectorizer.transform([text])
            prediction = self.sentiment_model.predict(X)[0]
            probabilities = self.sentiment_model.predict_proba(X)[0]
            confidence = np.max(probabilities)
            sentiment = "positive" if prediction == 1 else "negative"
            return {"sentiment": sentiment, "confidence": confidence}
        except:
            return {"sentiment": "unknown", "confidence": 0.0}

    def chat(self, message: str, use_neural: bool = True) -> str:
        cleaned_message = self.preprocessor.clean_text(message, level='medium')
        self.update_conversation_history(message)

        try:
            if use_neural and self.model:
                context = self.get_conversation_context()
                full_prompt = f"{context} {cleaned_message}".strip()
                response = self.generate_neural_response(full_prompt)
            else:
                response = self.generate_ngram_response(cleaned_message)

            response = self.preprocessor.clean_text(response, level='light')
            if not response or len(response.strip()) < 3:
                response = random.choice(self.config.fallback_responses)
            self.conversation_history.append(f"AI: {response}")
            self.train_ngram(f"{cleaned_message} {response}")
            return response

        except Exception as e:
            logging.error(f"Error in chat: {e}")
            return random.choice(self.config.fallback_responses)

    def update_conversation_history(self, message: str):
        with self.lock:
            self.conversation_history.append(f"User: {message}")
            if len(self.conversation_history) > self.config.context_window:
                self.conversation_history = self.conversation_history[-self.config.context_window:]

    def get_conversation_context(self) -> str:
        return " ".join(self.conversation_history[-20:])

    def save_model(self):
        try:
            model_data = {
                'config': self.config.__dict__,
                'unigram': dict(self.unigram),
                'bigram': {k: dict(v) for k, v in self.bigram.items()},
                'trigram': self._serialize_ngram(self.trigram),
                'ngram4': self._serialize_ngram(self.ngram4),
                'total_trigrams': self.total_trigrams,
                'record_counter': self.record_counter,
                'personality_bias': self.personality_bias,
                'training_losses': self.training_losses,
                'validation_losses': self.validation_losses,
                'tokenizer_word2idx': self.tokenizer.word2idx,
                'tokenizer_idx2word': self.tokenizer.idx2word,
                'tokenizer_word_freq': dict(self.tokenizer.word_freq)
            }
            if self.model:
                neural_path = self.model_file.replace('.pkl', '_neural.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'config': self.config.__dict__
                }, neural_path)

            if self.sentiment_model:
                sentiment_path = self.model_file.replace('.pkl', '_sentiment.pkl')
                with open(sentiment_path, 'wb') as f:
                    pickle.dump({
                        'model': self.sentiment_model,
                        'vectorizer': self.sentiment_vectorizer
                    }, f)

            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("Enhanced model saved successfully!")

        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)

                if 'config' in model_data:
                    config_dict = model_data['config']
                    self.config = ModelConfig(**config_dict)

                self.unigram = Counter(model_data.get('unigram', {}))
                self.bigram = defaultdict(Counter, {
                    k: Counter(v) for k, v in model_data.get('bigram', {}).items()
                })
                self.trigram = defaultdict(Counter, self._deserialize_ngram(model_data.get('trigram', {})))
                self.ngram4 = defaultdict(Counter, self._deserialize_ngram(model_data.get('ngram4', {})))
                self.total_trigrams = model_data.get('total_trigrams', 0)
                self.record_counter = model_data.get('record_counter', 0)
                self.personality_bias = model_data.get('personality_bias', {})
                self.training_losses = model_data.get('training_losses', [])
                self.validation_losses = model_data.get('validation_losses', [])

                if 'tokenizer_word2idx' in model_data:
                    self.tokenizer.word2idx = model_data['tokenizer_word2idx']
                    self.tokenizer.idx2word = model_data['tokenizer_idx2word']
                    self.tokenizer.word_freq = Counter(model_data.get('tokenizer_word_freq', {}))

                neural_path = self.model_file.replace('.pkl', '_neural.pth')
                if os.path.exists(neural_path):
                    self.initialize_neural_model()
                    checkpoint = torch.load(neural_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if checkpoint.get('optimizer_state_dict') and self.optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if checkpoint.get('scheduler_state_dict') and self.scheduler:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info("Neural model loaded successfully!")

                sentiment_path = self.model_file.replace('.pkl', '_sentiment.pkl')
                if os.path.exists(sentiment_path):
                    with open(sentiment_path, 'rb') as f:
                        sentiment_data = pickle.load(f)
                        self.sentiment_model = sentiment_data['model']
                        self.sentiment_vectorizer = sentiment_data['vectorizer']
                    logging.info("Sentiment model loaded successfully!")
                logging.info("Enhanced model loaded successfully!")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                self._initialize_empty_model()
        else:
            self._initialize_empty_model()

    def _initialize_empty_model(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0
        logging.info("Initialized empty model")

    def _serialize_ngram(self, ngram_dict: Dict) -> Dict:
        serialized = {}
        for key, counter_obj in ngram_dict.items():
            if isinstance(key, tuple):
                key_str = "|||".join(str(k) for k in key)
                serialized[key_str] = dict(counter_obj)
        return serialized

    def _deserialize_ngram(self, data: Dict) -> Dict:
        ngram = {}
        for key_str, value in data.items():
            key = tuple(key_str.split("|||"))
            ngram[key] = Counter(value)
        return ngram

    def get_training_stats(self) -> Dict:
        stats = {
            'total_records': self.record_counter,
            'vocabulary_size': len(self.unigram),
            'unique_bigrams': len(self.bigram),
            'unique_trigrams': len(self.trigram),
            'unique_4grams': len(self.ngram4),
            'total_trigrams': self.total_trigrams,
            'neural_model_loaded': self.model is not None,
            'sentiment_model_loaded': self.sentiment_model is not None,
            'device': str(self.device),
            'recent_training_losses': self.training_losses[-10:] if self.training_losses else [],
            'recent_validation_losses': self.validation_losses[-10:] if self.validation_losses else []
        }
        if self.unigram:
            stats['most_common_words'] = self.unigram.most_common(15)
        if self.model:
            param_count = sum(p.numel() for p in self.model.parameters())
            stats['neural_parameters'] = param_count

        return stats

    def set_personality(self, personality_traits: Dict[str, float]):
        with self.lock:
            self.personality_bias.update(personality_traits)
            logging.info(f"Updated personality with {len(personality_traits)} traits")

    def analyze_conversation_patterns(self) -> Dict:
        if not self.conversation_history:
            return {"message": "No conversation history available"}

        user_messages = [msg[5:] for msg in self.conversation_history if msg.startswith("User:")]
        ai_messages = [msg[3:] for msg in self.conversation_history if msg.startswith("AI:")]
        analysis = {
            'total_exchanges': len(user_messages),
            'avg_user_message_length': np.mean([len(msg.split()) for msg in user_messages]) if user_messages else 0,
            'avg_ai_message_length': np.mean([len(msg.split()) for msg in ai_messages]) if ai_messages else 0,
            'user_question_ratio': sum(1 for msg in user_messages if '?' in msg) / len(user_messages) if user_messages else 0,
            'conversation_topics': self._extract_topics(user_messages + ai_messages),
            'sentiment_distribution': self._analyze_conversation_sentiment(user_messages)
        }

        return analysis

    def _extract_topics(self, messages: List[str], top_k: int = 10) -> List[str]:
        if not messages:
            return []

        all_text = " ".join(messages).lower()
        tokens = word_tokenize(all_text)
        meaningful_tokens = [
            token for token in tokens 
            if len(token) > 3 and token.isalpha() and token not in self.preprocessor.stop_words
        ]

        token_freq = Counter(meaningful_tokens)
        return [word for word, _ in token_freq.most_common(top_k)]

    def _analyze_conversation_sentiment(self, messages: List[str]) -> Dict:
        if not messages or not self.sentiment_model:
            return {"positive": 0, "negative": 0, "unknown": len(messages)}

        sentiments = {"positive": 0, "negative": 0, "unknown": 0}
        for msg in messages:
            result = self.predict_sentiment(msg)
            sentiments[result["sentiment"]] += 1

        return sentiments

class AdvancedTrainer:
    def __init__(self, chatbot: EnhancedNeuralChat, max_workers: int = 8):
        self.chatbot = chatbot
        self.training_data = []
        self.sentiment_data = []
        self.max_workers = max_workers
        self.dataset_loader = MultiDatasetLoader(max_workers)

    def load_comprehensive_datasets(self, max_samples_per_dataset: int = 15000):
        available_datasets = [
            'openassistant',
            'dialogsum', 
            'empathetic',
            'persona_chat',
            'blended_skill',
            'wizard_of_wikipedia'
        ]

        logging.info(f"Loading {len(available_datasets)} datasets...")
        texts, sentiment_data = self.dataset_loader.load_multiple_datasets(
            available_datasets, 
            max_samples_per_dataset
        )
        self.training_data.extend(texts)
        self.sentiment_data.extend(sentiment_data)

        logging.info(f"Total loaded: {len(texts)} training samples, {len(sentiment_data)} sentiment samples")
        return len(texts)

    def parallel_ngram_training(self, batch_size: int = 1000):
        if not self.training_data:
            logging.warning("No training data available")
            return

        def process_batch(batch_texts):
            local_results = []
            for text in batch_texts:
                try:
                    cleaned_text = self.chatbot.preprocessor.clean_text(text, level='medium')
                    if cleaned_text and len(cleaned_text.strip()) > 5:
                        local_results.append(cleaned_text)
                except Exception as e:
                    continue
            return local_results

        batches = [
            self.training_data[i:i + batch_size] 
            for i in range(0, len(self.training_data), batch_size)
        ]
        logging.info(f"Processing {len(batches)} batches with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            processed_texts = []
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    processed_texts.extend(batch_results)
                except Exception as e:
                    logging.warning(f"Error processing batch: {e}")

        logging.info("Training n-gram models...")
        for i, text in enumerate(tqdm(processed_texts, desc="N-gram training")):
            try:
                self.chatbot.train_ngram(text)
                if i % 5000 == 0 and i > 0:
                    self.chatbot.save_model()
                    logging.info(f"Intermediate save at {i} samples")
            except Exception as e:
                continue

    def train_comprehensive_model(self, neural_epochs: int = 8, save_interval: int = 2000):
        if not self.training_data:
            logging.warning("No training data available. Load data first.")
            return

        logging.info("Starting comprehensive model training...")
        logging.info("Phase 1: N-gram model training...")
        self.parallel_ngram_training()

        neural_success = False
        if len(self.training_data) > 200:
            logging.info("Phase 2: Neural model training...")
            try:
                filtered_texts = self._filter_neural_training_data()
                if len(filtered_texts) > 100:
                    self.chatbot.train_neural_model(filtered_texts, epochs=neural_epochs)
                    neural_success = True
                    logging.info("Neural model training completed successfully!")
                else:
                    logging.warning("Insufficient quality data for neural training")
            except Exception as e:
                logging.error(f"Neural model training failed: {e}")
                logging.info("Continuing with N-gram model only...")
        else:
            logging.info("Insufficient data for neural training, using N-gram only")

        sentiment_success = False
        if self.sentiment_data and len(self.sentiment_data) > 100:
            logging.info("Phase 3: Sentiment analysis model training...")
            try:
                texts, labels = zip(*self.sentiment_data)
                self.chatbot.train_sentiment_model(list(texts), list(labels))
                sentiment_success = True
                logging.info("Sentiment model training completed successfully!")
            except Exception as e:
                logging.error(f"Sentiment model training failed: {e}")
        else:
            logging.info("Insufficient sentiment data, skipping sentiment training")

        try:
            self.chatbot.save_model()
            logging.info("Training completed!")
            stats = self.chatbot.get_training_stats()
            self._display_training_summary(stats, neural_success, sentiment_success)

        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def _filter_neural_training_data(self) -> List[str]:
        filtered_texts = []

        for text in self.training_data:
            if (10 <= len(text.split()) <= 200 and
                len(text.strip()) > 15 and
                not text.lower().startswith('http') and
                sum(c.isalpha() for c in text) > len(text) * 0.5):
                filtered_texts.append(text)
        logging.info(f"Filtered {len(filtered_texts)} high-quality texts from {len(self.training_data)} total")
        return filtered_texts

    def _display_training_summary(self, stats: Dict, neural_success: bool, sentiment_success: bool):
        logging.info("\n" + "="*60)
        logging.info("TRAINING SUMMARY")
        logging.info("="*60)
        logging.info(f"✓ N-gram training: SUCCESS")
        logging.info(f"{'✓' if neural_success else '✗'} Neural training: {'SUCCESS' if neural_success else 'FAILED/SKIPPED'}")
        logging.info(f"{'✓' if sentiment_success else '✗'} Sentiment training: {'SUCCESS' if sentiment_success else 'FAILED/SKIPPED'}")
        logging.info("\nModel Statistics:")
        for key, value in stats.items():
            if key not in ['recent_training_losses', 'recent_validation_losses', 'most_common_words']:
                logging.info(f"  {key}: {value}")
        if stats.get('most_common_words'):
            logging.info(f"\nMost common words: {[word for word, _ in stats['most_common_words'][:10]]}")
        logging.info("="*60)

    def interactive_training_session(self):
        logging.info("Starting enhanced interactive training session...")
        logging.info("Commands: 'quit', 'stats', 'save', 'sentiment <text>', 'personality <trait> <value>'")
        logging.info("          'neural', 'ngram', 'analyze', 'help'")
        session_data = {
            'interactions': 0,
            'positive_feedback': 0,
            'negative_feedback': 0
        }

        while True:
            try:
                print("\n" + "-"*50)
                user_input = input("You: ").strip()
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_interactive_stats(session_data)
                    continue
                elif user_input.lower() == 'save':
                    self.chatbot.save_model()
                    print("✓ Model saved!")
                    continue
                elif user_input.lower() == 'analyze':
                    analysis = self.chatbot.analyze_conversation_patterns()
                    self._display_analysis(analysis)
                    continue
                elif user_input.startswith('sentiment '):
                    text = user_input[10:]
                    result = self.chatbot.predict_sentiment(text)
                    print(f"Sentiment: {result}")
                    continue
                elif user_input.startswith('personality '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        trait, value = parts[1], float(parts[2])
                        self.chatbot.set_personality({trait: value})
                        print(f"✓ Set personality trait: {trait} = {value}")
                    continue
                elif not user_input:
                    continue

                mode = input("Mode (neural/ngram/auto): ").strip().lower()
                if mode == 'neural':
                    use_neural = True
                elif mode == 'ngram':
                    use_neural = False
                else:  # auto
                    use_neural = self.chatbot.model is not None

                start_time = time.time()
                response = self.chatbot.chat(user_input, use_neural=use_neural)
                response_time = time.time() - start_time

                print(f"\nAI ({mode if mode in ['neural', 'ngram'] else 'auto'}): {response}")
                print(f"Response time: {response_time:.3f}s")
                feedback = input("\nRate response (1-5) or 'skip': ").strip()
                if feedback.isdigit() and 1 <= int(feedback) <= 5:
                    rating = int(feedback)
                    session_data['interactions'] += 1
                    if rating >= 4:
                        session_data['positive_feedback'] += 1
                        self.chatbot.train_ngram(f"{user_input} {response}")
                        print("✓ Positive feedback recorded")
                    elif rating <= 2:
                        session_data['negative_feedback'] += 1
                        print("✗ Negative feedback recorded")

                sentiment_result = self.chatbot.predict_sentiment(user_input)
                if sentiment_result["sentiment"] != "unknown":
                    print(f"Your sentiment: {sentiment_result['sentiment']} "
                          f"(confidence: {sentiment_result['confidence']:.2f})")

            except KeyboardInterrupt:
                print("\nExiting interactive session...")
                break
            except Exception as e:
                logging.error(f"Error in interactive session: {e}")
                print(f"Error: {e}")
                continue

        self._show_session_summary(session_data)
        self.chatbot.save_model()
        print("Interactive training session ended. Model saved.")

    def _show_help(self):
        help_text = """
Available Commands:
  quit                        - Exit the session
  stats                       - Show model statistics
  save                        - Save current model
  analyze                     - Analyze conversation patterns
  sentiment <text>            - Analyze sentiment of text
  personality <trait> <value> - Set personality trait
  neural                      - Use neural model for next response
  ngram                       - Use n-gram model for next response
  help                        - Show this help message

Response Modes:
  neural - Use neural network model (if available)
  ngram  - Use n-gram statistical model
  auto   - Automatically choose best available model

Rating System:
  1-2: Negative feedback (model learns to avoid similar responses)
  3:   Neutral feedback (no learning impact)
  4-5: Positive feedback (model reinforces similar responses)
"""
        print(help_text)

    def _show_interactive_stats(self, session_data: Dict):
        stats = self.chatbot.get_training_stats()
        print("\n" + "="*40)
        print("SESSION STATISTICS")
        print("="*40)
        print(f"Interactions: {session_data['interactions']}")
        print(f"Positive feedback: {session_data['positive_feedback']}")
        print(f"Negative feedback: {session_data['negative_feedback']}")
        if session_data['interactions'] > 0:
            satisfaction = session_data['positive_feedback'] / session_data['interactions'] * 100
            print(f"Satisfaction rate: {satisfaction:.1f}%")
        print("\nMODEL STATISTICS")
        print("-"*20)
        for key, value in stats.items():
            if key not in ['recent_training_losses', 'recent_validation_losses', 'most_common_words']:
                print(f"{key}: {value}")

    def _display_analysis(self, analysis: Dict):
        print("\n" + "="*40)
        print("CONVERSATION ANALYSIS") 
        print("="*40)
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            elif isinstance(value, list):
                print(f"{key}: {', '.join(map(str, value[:5]))}")
            else:
                print(f"{key}: {value}")

    def _show_session_summary(self, session_data: Dict):
        print("\n" + "="*40)
        print("SESSION SUMMARY")
        print("="*40)
        print(f"Total interactions: {session_data['interactions']}")
        print(f"Positive feedback: {session_data['positive_feedback']}")
        print(f"Negative feedback: {session_data['negative_feedback']}")
        if session_data['interactions'] > 0:
            satisfaction = session_data['positive_feedback'] / session_data['interactions'] * 100
            print(f"Overall satisfaction: {satisfaction:.1f}%")

    def benchmark_performance(self, test_inputs: List[str] = None):
        if test_inputs is None:
            test_inputs = [
                "Hello, how are you today?",
                "What's the weather like?",
                "Can you help me with programming?",
                "Tell me a joke please",
                "What do you think about artificial intelligence?",
                "How do I cook pasta?",
                "What's the meaning of life?",
                "Can you explain quantum physics?",
                "What's your favorite color?",
                "How do I learn machine learning?",
                "I'm feeling sad today",
                "That's really exciting news!",
                "I don't understand this concept",
                "Thank you for your help",
                "This is frustrating me"
            ]

        logging.info("Starting comprehensive performance benchmark...")
        results = []

        for test_input in tqdm(test_inputs, desc="Benchmarking"):
            result = {
                'input': test_input,
                'input_length': len(test_input.split())
            }
            if self.chatbot.model:
                start_time = time.time()
                neural_response = self.chatbot.generate_neural_response(test_input)
                neural_time = time.time() - start_time
                result.update({
                    'neural_response': neural_response,
                    'neural_time': neural_time,
                    'neural_length': len(neural_response.split()),
                    'neural_quality': self._assess_response_quality(neural_response)
                })

            start_time = time.time()
            ngram_response = self.chatbot.generate_ngram_response(test_input)
            ngram_time = time.time() - start_time
            result.update({
                'ngram_response': ngram_response,
                'ngram_time': ngram_time,
                'ngram_length': len(ngram_response.split()),
                'ngram_quality': self._assess_response_quality(ngram_response)
            })

            sentiment = self.chatbot.predict_sentiment(test_input)
            result['input_sentiment'] = sentiment
            results.append(result)

        self._display_benchmark_results(results)
        return results

    def _assess_response_quality(self, response: str) -> Dict:
        if not response:
            return {'score': 0, 'issues': ['empty_response']}

        issues = []
        score = 100

        word_count = len(response.split())
        if word_count < 3:
            issues.append('too_short')
            score -= 300
        elif word_count > 1000:
            issues.append('too_long')
            score -= 10
        words = response.lower().split()
        if len(set(words)) < len(words) * 0.7:
            issues.append('repetitive')
            score -= 20
        if response.count('.') == 0 and word_count > 15:
            issues.append('no_punctuation')
            score -= 10
        if response.islower() or response.isupper():
            issues.append('capitalization')
            score -= 5

        return {'score': max(0, score), 'issues': issues}

    def _display_benchmark_results(self, results: List[Dict]):
        logging.info("\n" + "="*60)
        logging.info("BENCHMARK RESULTS")
        logging.info("="*60)

        if any('neural_time' in r for r in results):
            neural_times = [r['neural_time'] for r in results if 'neural_time' in r]
            neural_lengths = [r['neural_length'] for r in results if 'neural_length' in r]
            neural_qualities = [r['neural_quality']['score'] for r in results if 'neural_quality' in r]
            logging.info("Neural Model Performance:")
            logging.info(f"  Average response time: {np.mean(neural_times):.4f}s")
            logging.info(f"  Average response length: {np.mean(neural_lengths):.1f} words")
            logging.info(f"  Average quality score: {np.mean(neural_qualities):.1f}/100")

        ngram_times = [r['ngram_time'] for r in results]
        ngram_lengths = [r['ngram_length'] for r in results]
        ngram_qualities = [r['ngram_quality']['score'] for r in results]
        logging.info("N-gram Model Performance:")
        logging.info(f"  Average response time: {np.mean(ngram_times):.4f}s")
        logging.info(f"  Average response length: {np.mean(ngram_lengths):.1f} words")
        logging.info(f"  Average quality score: {np.mean(ngram_qualities):.1f}/100")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'benchmark_results_{timestamp}.json'

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Detailed results saved to: {results_file}")
        except Exception as e:
            logging.error(f"Error saving benchmark results: {e}")

def create_enhanced_chatbot(config_overrides: Dict = None) -> EnhancedNeuralChat:
    config = ModelConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return EnhancedNeuralChat(config)

def handle_interrupt(signal_received, frame, chatbot):
    global locker
    logging.info("Interrupt received! Saving model before exiting...")
    try:
        chatbot.save_model()
        logging.info("Model saved successfully!")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

    locker = True
    logging.info("Exiting gracefully...")
    exit(0)

def main():
    print("AI builder-xc v2")
    print("=" * 50)
    print("Available modes:")
    print("1. comprehensive - Full training with multiple datasets")
    print("2. interactive   - Interactive training session")
    print("3. test          - Test chat mode")
    print("4. benchmark     - Performance benchmark")
    print("5. analyze       - Analyze existing model")
    print("6. datasets      - Load specific datasets only")

    mode = input("\nChoose mode (1-6): ").strip()
    config_overrides = {
        'vocab_size': 25000,
        'embedding_dim': 384,
        'hidden_dim': 768,
        'num_layers': 4,
        'dropout': 0.15,
        'learning_rate': 0.0005,
        'batch_size': 48,
        'max_sequence_length': 256,
        'min_word_freq': 3
    }

    print("\n[/] Initializing enhanced chatbot...")
    chatbot = create_enhanced_chatbot(config_overrides)
    signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, chatbot))

    if mode == '1':
        print("\n[~] Starting comprehensive training mode...")
        trainer = AdvancedTrainer(chatbot, max_workers=8)

        load_datasets = input("Load multiple datasets? (y/n): ").strip().lower() == 'y'
        if load_datasets:
            max_samples = input("Max samples per dataset (default 15000): ").strip()
            max_samples = int(max_samples) if max_samples.isdigit() else 15000
            trainer.load_comprehensive_datasets(max_samples)

        epochs = input("Neural training epochs (default 8): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 8
        trainer.train_comprehensive_model(neural_epochs=epochs)

    elif mode == '2':
        print("\n[-] Starting interactive training mode...")
        trainer = AdvancedTrainer(chatbot)
        trainer.interactive_training_session()

    elif mode == '3':
        print("\nEntering test chat mode...")
        print("Commands: 'exit', 'stats', 'sentiment <text>', 'personality <trait> <value>'")
        print("          'analyze', 'save', 'help'")
        while True:
            try:
                user_input = input("\nQuestion: ").strip()

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    print("Available commands: exit, stats, sentiment <text>, personality <trait> <value>, analyze, save")
                    continue
                elif user_input.lower() == 'stats':
                    stats = chatbot.get_training_stats()
                    for key, value in stats.items():
                        if key not in ['recent_training_losses', 'recent_validation_losses', 'most_common_words']:
                            print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'analyze':
                    analysis = chatbot.analyze_conversation_patterns()
                    for key, value in analysis.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'save':
                    chatbot.save_model()
                    print("Model saved!")
                    continue
                elif user_input.startswith('sentiment '):
                    text = user_input[10:]
                    result = chatbot.predict_sentiment(text)
                    print(f"Sentiment: {result}")
                    continue
                elif user_input.startswith('personality '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        trait, value = parts[1], float(parts[2])
                        chatbot.set_personality({trait: value})
                        print(f"Set personality trait: {trait} = {value}")
                    continue
                elif not user_input:
                    continue
                use_neural = chatbot.model is not None
                start_time = time.time()
                response = chatbot.chat(user_input, use_neural=use_neural)
                response_time = time.time() - start_time

                print(f"AI: {response}")
                print(f"Response time: {response_time:.3f}s")
                sentiment = chatbot.predict_sentiment(user_input)
                if sentiment['sentiment'] != 'unknown':
                    print(f"Your sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.2f})")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    elif mode == '4':
        print("\n{=] Starting performance benchmark...")
        trainer = AdvancedTrainer(chatbot)
        trainer.benchmark_performance()

    elif mode == '5':
        print("\n[-] Analyzing existing model...")
        stats = chatbot.get_training_stats()
        patterns = chatbot.analyze_conversation_patterns()

        print("\nModel Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            if key not in ['recent_training_losses', 'recent_validation_losses', 'most_common_words']:
                print(f"{key}: {value}")
        if stats.get('most_common_words'):
            print(f"\nMost common words: {[word for word, _ in stats['most_common_words'][:10]]}")
        print("\nConversation Patterns:")
        print("-" * 30)
        for key, value in patterns.items():
            print(f"{key}: {value}")

    elif mode == '6':
        print("\n[~] Dataset loading mode...")
        trainer = AdvancedTrainer(chatbot)
        available_datasets = ['openassistant', 'dialogsum', 'empathetic', 'persona_chat', 'blended_skill', 'wizard_of_wikipedia']
        print(f"Available datasets: {', '.join(available_datasets)}")
        selected = input("Enter dataset names (comma-separated): ").strip().split(',')
        selected = [name.strip() for name in selected if name.strip() in available_datasets]

        if selected:
            max_samples = input("Max samples per dataset (default 10000): ").strip()
            max_samples = int(max_samples) if max_samples.isdigit() else 10000
            texts, sentiment_data = trainer.dataset_loader.load_multiple_datasets(selected, max_samples)
            trainer.training_data.extend(texts)
            trainer.sentiment_data.extend(sentiment_data)

            print(f"[*] Loaded {len(texts)} texts and {len(sentiment_data)} sentiment samples")
            if input("Train on loaded data? (y/n): ").strip().lower() == 'y':
                trainer.parallel_ngram_training()
    else:
        print("[+] Invalid mode selected!")
        return

    print("\n[+] Saving model...")
    chatbot.save_model()
    print("[+] Session completed successfully!")

if __name__ == "__main__":
    main()
