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
import math

locker = False
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ModelConfig:
    vocab_size: int = 15000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    max_sequence_length: int = 128
    softmax_temperature: float = 0.7
    weight_factor: float = 1.2
    top_k: int = 40
    top_p: float = 0.9
    context_window: int = 100
    response_length_factor: float = 1.0

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.lemmatizer = None

    def clean_text(self, text: str, level: str = 'medium') -> str:
        if not text:
            return ""

        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        if level in ['medium', 'heavy']:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            text = text.lower().strip()

        if level == 'heavy' and self.lemmatizer:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
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
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

class ImprovedLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            config.embedding_dim, 
            config.hidden_dim, 
            config.num_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            config.hidden_dim * 2, 
            num_heads=8, 
            dropout=config.dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.output_projection = nn.Linear(config.hidden_dim * 2, config.vocab_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and param.dim() == 1:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        attended, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attention_mask)
        attended = self.layer_norm(attended + lstm_out)

        output = self.output_projection(attended)
        return output

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(tokens, dtype=torch.long)

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 15000):
        self.vocab_size = vocab_size
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word_freq = Counter()

    def build_vocab(self, texts: List[str]):
        for text in texts:
            tokens = word_tokenize(text.lower())
            self.word_freq.update(tokens)

        most_common = self.word_freq.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_length: int = 128) -> List[int]:
        tokens = word_tokenize(text.lower())
        indices = [self.word2idx.get(token, 1) for token in tokens]  # 1 is <unk>
        if len(indices) > max_length - 2:
            indices = indices[:max_length-2]

        indices = [2] + indices + [3]  # <sos> + tokens + <eos>

        while len(indices) < max_length:
            indices.append(0)

        return indices

    def decode(self, indices: List[int]) -> str:
        tokens = [self.idx2word.get(idx, '<unk>') for idx in indices]
        tokens = [t for t in tokens if t not in ['<pad>', '<sos>', '<eos>']]
        return " ".join(tokens)

class EnhancedNeuralChat:
    def __init__(self, config: ModelConfig = None, model_file: str = 'enhanced_ai_model.pkl'):
        self.config = config or ModelConfig()
        self.model_file = model_file
        self.preprocessor = TextPreprocessor()
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)

        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conversation_history = []
        self.personality_bias = {}
        self.lock = threading.Lock()
        self.sentiment_model = None
        self.sentiment_vectorizer = TfidfVectorizer(max_features=5000)

        self.training_losses = []
        self.validation_losses = []

        self.load_model()

    def initialize_neural_model(self):
        self.model = ImprovedLSTM(self.config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        logging.info(f"Neural model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train_neural_model(self, texts: List[str], epochs: int = 5, validation_split: float = 0.1):
        if not self.model:
            try:
                self.initialize_neural_model()
            except Exception as e:
                logging.error(f"Failed to initialize neural model: {e}")
                logging.info("Falling back to n-gram only training")
                return

        if len(texts) < 10:
            logging.warning("Not enough training data for neural model. Skipping neural training.")
            return

        logging.info("Building vocabulary...")
        self.tokenizer.build_vocab(texts)
        if len(self.tokenizer.word2idx) < 100:
            logging.warning("Vocabulary too small for effective neural training. Skipping.")
            return

        actual_vocab_size = len(self.tokenizer.word2idx)
        if actual_vocab_size != self.config.vocab_size:
            logging.info(f"Updating vocab size from {self.config.vocab_size} to {actual_vocab_size}")
            self.config.vocab_size = actual_vocab_size
            self.model = ImprovedLSTM(self.config).to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
        train_texts, val_texts = train_test_split(texts, test_size=validation_split, random_state=42)

        try:
            train_dataset = TextDataset(train_texts, self.tokenizer, self.config.max_sequence_length)
            val_dataset = TextDataset(val_texts, self.tokenizer, self.config.max_sequence_length)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        except Exception as e:
            logging.error(f"Failed to create data loaders: {e}")
            return

        self.model.train()
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            try:
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                for batch in progress_bar:
                    try:
                        batch = batch.to(self.device)
                        if batch.size(0) < 2:
                            continue

                        input_ids = batch[:, :-1]
                        targets = batch[:, 1:]
                        if input_ids.size(1) < 2:
                            continue

                        self.optimizer.zero_grad()
                        outputs = self.model(input_ids)
                        loss = F.cross_entropy(
                            outputs.reshape(-1, self.config.vocab_size),
                            targets.reshape(-1),
                            ignore_index=0
                        )

                        if torch.isnan(loss):
                            logging.warning("NaN loss detected, skipping batch")
                            continue

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1
                        progress_bar.set_postfix({'loss': loss.item()})
                    except Exception as batch_e:
                        logging.warning(f"Error in batch processing: {batch_e}")
                        continue
                if num_batches == 0:
                    logging.error("No valid batches processed")
                    break

                avg_train_loss = total_loss / num_batches
                val_loss = self.validate_neural_model(val_loader)
                self.training_losses.append(avg_train_loss)
                self.validation_losses.append(val_loss)
                logging.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping after {epoch+1} epochs")
                        break

            except Exception as epoch_e:
                logging.error(f"Error in epoch {epoch+1}: {epoch_e}")
                break

    def validate_neural_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(self.device)

                    if batch.size(0) < 2:
                        continue

                    input_ids = batch[:, :-1]
                    targets = batch[:, 1:]

                    if input_ids.size(1) < 2:
                        continue

                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(
                        outputs.reshape(-1, self.config.vocab_size),
                        targets.reshape(-1),
                        ignore_index=0
                    )
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    logging.warning(f"Error in validation batch: {e}")
                    continue

        self.model.train()

        if num_batches == 0:
            logging.warning("No valid validation batches")
            return float('inf')

        return total_loss / num_batches

    def generate_neural_response(self, prompt: str, max_length: int = 50) -> str:
        if not self.model:
            return self.generate_ngram_response(prompt)

        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt, max_length=self.config.max_sequence_length)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            generated = tokens.copy()

            for _ in range(max_length):
                if len(generated) >= self.config.max_sequence_length:
                    break
                current_input = torch.tensor([generated[-self.config.max_sequence_length:]], 
                                           dtype=torch.long).to(self.device)
                outputs = self.model(current_input)
                logits = outputs[0, -1, :] / self.config.softmax_temperature
                next_token = self.sample_token(logits)
                if next_token == 3:  # <eos>
                    break
                generated.append(next_token)

            response = self.tokenizer.decode(generated)
            prompt_clean = self.preprocessor.clean_text(prompt)
            if response.startswith(prompt_clean):
                response = response[len(prompt_clean):].strip()

        self.model.train()
        return response

    def sample_token(self, logits: torch.Tensor) -> int:
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        return next_token

    def generate_ngram_response(self, prompt: str) -> str:
        tokens = word_tokenize(prompt.lower())
        response = []

        for n in [3, 2, 1]:
            if len(tokens) >= n:
                seed = tuple(tokens[-n:])
                if n == 3 and seed in self.ngram4 and sum(self.ngram4[seed].values()) > 0:
                    response.extend(list(seed))
                    next_token = self.top_k_top_p_sampling(self.ngram4[seed])
                    response.append(next_token)
                    break
                elif n == 2 and seed in self.trigram and sum(self.trigram[seed].values()) > 0:
                    response.extend(list(seed))
                    next_token = self.top_k_top_p_sampling(self.trigram[seed])
                    response.append(next_token)
                    break
                elif n == 1 and seed[0] in self.bigram and sum(self.bigram[seed[0]].values()) > 0:
                    response.append(seed[0])
                    next_token = self.top_k_top_p_sampling(self.bigram[seed[0]])
                    response.append(next_token)
                    break
        if not response and self.unigram:
            response.append(random.choice(list(self.unigram.keys())))

        desired_length = random.randint(20, 50)
        while len(response) < desired_length:
            next_token = None
            if len(response) >= 3:
                key4 = tuple(response[-3:])
                if key4 in self.ngram4 and sum(self.ngram4[key4].values()) > 0:
                    next_token = self.top_k_top_p_sampling(self.ngram4[key4])
            if not next_token and len(response) >= 2:
                key3 = (response[-2], response[-1])
                if key3 in self.trigram and sum(self.trigram[key3].values()) > 0:
                    next_token = self.top_k_top_p_sampling(self.trigram[key3])
            if not next_token and response and response[-1] in self.bigram:
                if sum(self.bigram[response[-1]].values()) > 0:
                    next_token = self.top_k_top_p_sampling(self.bigram[response[-1]])
            if not next_token:
                break
            response.append(next_token)
        return " ".join(response)

    def top_k_top_p_sampling(self, counter: Counter) -> str:
        if not counter:
            return ""

        biased_counter = {}
        for word, count in counter.items():
            bias = self.personality_bias.get(word, 1.0)
            biased_counter[word] = count * bias
        words = list(biased_counter.keys())
        counts = np.array(list(biased_counter.values()), dtype=float)
        logits = np.log(counts * self.config.weight_factor + 1e-10)

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
            self.unigram.update(tokens)
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])
            self.record_counter += 1

    def train_sentiment_model(self, texts: List[str], labels: List[int]):
        if len(texts) != len(labels):
            logging.warning("Texts and labels length mismatch for sentiment training")
            return
        X = self.sentiment_vectorizer.fit_transform(texts)
        y = np.array(labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
        self.sentiment_model.fit(X_train, y_train)

        y_pred = self.sentiment_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info(f"Sentiment model trained with accuracy: {accuracy:.4f}")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_val, y_pred))

    def predict_sentiment(self, text: str) -> Dict:
        if not self.sentiment_model:
            return {"sentiment": "unknown", "confidence": 0.0}
        X = self.sentiment_vectorizer.transform([text])
        prediction = self.sentiment_model.predict(X)[0]
        probabilities = self.sentiment_model.predict_proba(X)[0]
        confidence = np.max(probabilities)

        sentiment = "positive" if prediction == 1 else "negative"
        return {"sentiment": sentiment, "confidence": confidence}

    def chat(self, message: str, use_neural: bool = True) -> str:
        cleaned_message = self.preprocessor.clean_text(message, level='medium')
        self.update_conversation_history(message)

        if use_neural and self.model:
            context = self.get_conversation_context()
            full_prompt = f"{context} {cleaned_message}".strip()
            response = self.generate_neural_response(full_prompt)
        else:
            response = self.generate_ngram_response(cleaned_message)
        response = self.preprocessor.clean_text(response, level='light')
        self.conversation_history.append(f"AI: {response}")
        self.train_ngram(f"{cleaned_message} {response}")
        return response

    def update_conversation_history(self, message: str):
        self.conversation_history.append(f"User: {message}")
        if len(self.conversation_history) > self.config.context_window:
            self.conversation_history = self.conversation_history[-self.config.context_window:]

    def get_conversation_context(self) -> str:
        return " ".join(self.conversation_history[-10:])

    def save_model(self):
        """Enhanced model saving"""
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
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
            }, self.model_file.replace('.pkl', '_neural.pth'))
        if self.sentiment_model:
            with open(self.model_file.replace('.pkl', '_sentiment.pkl'), 'wb') as f:
                pickle.dump({
                    'model': self.sentiment_model,
                    'vectorizer': self.sentiment_vectorizer
                }, f)
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info("Enhanced model saved successfully!")

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)

                if 'config' in model_data:
                    config_dict = model_data['config']
                    self.config = ModelConfig(**config_dict)

                self.unigram = Counter(model_data.get('unigram', {}))
                self.bigram = defaultdict(Counter, {k: Counter(v) for k, v in model_data.get('bigram', {}).items()})
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
                    if checkpoint['optimizer_state_dict'] and self.optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            key_str = "|||".join(key)
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
            'training_losses': self.training_losses[-10:] if self.training_losses else [],
            'validation_losses': self.validation_losses[-10:] if self.validation_losses else []
        }
        if self.unigram:
            stats['most_common_words'] = self.unigram.most_common(10)
        return stats

    def set_personality(self, personality_traits: Dict[str, float]):
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
            'conversation_topics': self._extract_topics(user_messages + ai_messages)
        }

        return analysis

    def _extract_topics(self, messages: List[str], top_k: int = 5) -> List[str]:
        if not messages:
            return []
        all_text = " ".join(messages).lower()
        tokens = word_tokenize(all_text)
        meaningful_tokens = [token for token in tokens if len(token) > 3 and token.isalpha()]
        token_freq = Counter(meaningful_tokens)

        return [word for word, _ in token_freq.most_common(top_k)]

class AdvancedTrainer:
    def __init__(self, chatbot: EnhancedNeuralChat):
        self.chatbot = chatbot
        self.training_data = []
        self.sentiment_data = []

    def load_external_dataset(self, dataset_name: str = "OpenAssistant/oasst1",
                            max_samples: int = 50000, batch_size: int = 32):
        try:
            from datasets import load_dataset
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logging.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")
            valid_data = [row for row in dataset if row.get("text") and len(row["text"].strip()) > 10]
            if max_samples:
                valid_data = valid_data[:max_samples]
            logging.info(f"Processing {len(valid_data)} samples...")

            texts_for_training = []
            texts_for_sentiment = []
            labels_for_sentiment = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(self._process_sample, row): idx
                    for idx, row in enumerate(valid_data)
                }
                processed_count = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                    try:
                        result = future.result()
                        if result:
                            processed_text, sentiment_label = result
                            texts_for_training.append(processed_text)
                            if sentiment_label is not None:
                                texts_for_sentiment.append(processed_text)
                                labels_for_sentiment.append(sentiment_label)
                        processed_count += 1
                        if processed_count % 1000 == 0:
                            logging.info(f"Processed {processed_count}/{len(valid_data)} samples")

                    except Exception as e:
                        logging.warning(f"Error processing sample: {e}")
                        continue

            self.training_data.extend(texts_for_training)
            self.sentiment_data.extend(list(zip(texts_for_sentiment, labels_for_sentiment)))
            logging.info(f"Successfully loaded {len(texts_for_training)} training samples")
            logging.info(f"Successfully loaded {len(texts_for_sentiment)} sentiment samples")
            return len(texts_for_training)

        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            return 0

    def _process_sample(self, row: Dict) -> Optional[Tuple[str, Optional[int]]]:
        """Process individual sample from dataset"""
        text = row.get("text", "").strip()
        if not text or len(text) < 10:
            return None

        cleaned_text = self.chatbot.preprocessor.clean_text(text, level='medium')
        if not cleaned_text:
            return None

        sentiment_label = None
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'like', 'enjoy']
        negative_indicators = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'problem', 'issue', 'error']

        text_lower = cleaned_text.lower()
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        if positive_count > negative_count:
            sentiment_label = 1
        elif negative_count > positive_count:
            sentiment_label = 0
        return cleaned_text, sentiment_label

    def train_comprehensive_model(self, neural_epochs: int = 5, 
                                save_interval: int = 1000):
        if not self.training_data:
            logging.warning("No training data available. Load data first.")
            return
        logging.info("Starting comprehensive model training...")
        logging.info("Training N-gram models...")
        for i, text in enumerate(tqdm(self.training_data, desc="N-gram training")):
            try:
                self.chatbot.train_ngram(text)
                if i % save_interval == 0 and i > 0:
                    self.chatbot.save_model()
                    logging.info(f"Intermediate save at {i} samples")
            except Exception as e:
                logging.warning(f"Error training on sample {i}: {e}")
                continue

        neural_training_success = False
        if len(self.training_data) > 100:
            logging.info("Training neural model...")
            try:
                self.chatbot.train_neural_model(self.training_data, epochs=neural_epochs)
                neural_training_success = True
                logging.info("Neural model training completed successfully!")
            except Exception as e:
                logging.error(f"Neural model training failed: {e}")
                logging.info("Continuing with N-gram model only...")
        else:
            logging.info("Insufficient data for neural training, using N-gram only")

        sentiment_training_success = False
        if self.sentiment_data and len(self.sentiment_data) > 50:
            logging.info("Training sentiment analysis model...")
            try:
                texts, labels = zip(*self.sentiment_data)
                self.chatbot.train_sentiment_model(list(texts), list(labels))
                sentiment_training_success = True
                logging.info("Sentiment model training completed successfully!")
            except Exception as e:
                logging.error(f"Sentiment model training failed: {e}")
                logging.info("Continuing without sentiment analysis...")
        else:
            logging.info("Insufficient sentiment data, skipping sentiment training")

        try:
            self.chatbot.save_model()
            logging.info("Comprehensive training completed!")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

        try:
            stats = self.chatbot.get_training_stats()
            logging.info("Training Statistics:")
            for key, value in stats.items():
                logging.info(f"  {key}: {value}")

            logging.info("Training Summary:")
            logging.info(f"  N-gram training: ✓ Success")
            logging.info(f"  Neural training: {'✓ Success' if neural_training_success else '✗ Failed/Skipped'}")
            logging.info(f"  Sentiment training: {'✓ Success' if sentiment_training_success else '✗ Failed/Skipped'}")

        except Exception as e:
            logging.error(f"Error displaying statistics: {e}")

    def interactive_training_session(self):
        logging.info("Starting interactive training session...")
        logging.info("Type 'quit' to exit, 'stats' for statistics, 'save' to save model")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = self.chatbot.get_training_stats()
                    print("\nTraining Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'save':
                    self.chatbot.save_model()
                    print("Model saved!")
                    continue
                elif not user_input:
                    continue
                response = self.chatbot.chat(user_input)
                print(f"AI: {response}")

                feedback = input("Rate this response (1-5, or 'skip'): ").strip()
                if feedback.isdigit() and 1 <= int(feedback) <= 5:
                    rating = int(feedback)
                    if rating >= 4:
                        self.chatbot.train_ngram(f"{user_input} {response}")
                        print("✓ Positive feedback recorded")
                    elif rating <= 2:
                        print("✗ Negative feedback recorded")
                sentiment_result = self.chatbot.predict_sentiment(user_input)
                if sentiment_result["sentiment"] != "unknown":
                    print(f"Detected sentiment: {sentiment_result['sentiment']} "
                          f"(confidence: {sentiment_result['confidence']:.2f})")

            except KeyboardInterrupt:
                print("\nExiting interactive session...")
                break
            except Exception as e:
                logging.error(f"Error in interactive session: {e}")
                continue

        self.chatbot.save_model()
        print("Interactive training session ended. Model saved.")

def create_enhanced_chatbot(config_overrides: Dict = None) -> EnhancedNeuralChat:
    config = ModelConfig()

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return EnhancedNeuralChat(config)

def handle_interrupt(signal_received, frame, chatbot):
    global locker
    logging.info("Saving model before exiting...")
    chatbot.save_model()
    locker = True
    logging.info("Model saved successfully!")
    exit(0)

def benchmark_model_performance(chatbot: EnhancedNeuralChat, test_inputs: List[str]):
    logging.info("Starting model performance benchmark...")
    results = []
    for test_input in tqdm(test_inputs, desc="Benchmarking"):
        start_time = time.time()

        neural_response = chatbot.generate_neural_response(test_input)
        neural_time = time.time() - start_time

        start_time = time.time()
        ngram_response = chatbot.generate_ngram_response(test_input)
        ngram_time = time.time() - start_time
        neural_features = chatbot.preprocessor.extract_features(neural_response)
        ngram_features = chatbot.preprocessor.extract_features(ngram_response)
        results.append({
            'input': test_input,
            'neural_response': neural_response,
            'ngram_response': ngram_response,
            'neural_time': neural_time,
            'ngram_time': ngram_time,
            'neural_length': neural_features['word_count'],
            'ngram_length': ngram_features['word_count']
        })

    avg_neural_time = np.mean([r['neural_time'] for r in results])
    avg_ngram_time = np.mean([r['ngram_time'] for r in results])
    avg_neural_length = np.mean([r['neural_length'] for r in results])
    avg_ngram_length = np.mean([r['ngram_length'] for r in results])
    logging.info("Benchmark Results:")
    logging.info(f"  Average Neural Response Time: {avg_neural_time:.4f}s")
    logging.info(f"  Average N-gram Response Time: {avg_ngram_time:.4f}s")
    logging.info(f"  Average Neural Response Length: {avg_neural_length:.1f} words")
    logging.info(f"  Average N-gram Response Length: {avg_ngram_length:.1f} words")

    return results

def main():
    print("Enhanced AI Model Trainer")
    print("=" * 40)
    print("Available modes:")
    print("1. train - Comprehensive training with external data")
    print("2. interactive - Interactive training session")
    print("3. test - Test chat mode")
    print("4. benchmark - Performance benchmark")
    print("5. analyze - Analyze existing model")
    mode = input("\nChoose mode (1-5): ").strip()
    config_overrides = {
        'vocab_size': 20000,
        'embedding_dim': 256,
        'hidden_dim': 512,
        'dropout': 0.1,
        'learning_rate': 0.001
    }
    chatbot = create_enhanced_chatbot(config_overrides)
    signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, chatbot))

    if mode == '1':
        trainer = AdvancedTrainer(chatbot)
        dataset_choice = input("Use OpenAssistant dataset? (y/n): ").strip().lower()
        if dataset_choice == 'y':
            max_samples = input("Max samples (default 50000): ").strip()
            max_samples = int(max_samples) if max_samples.isdigit() else 50000
            trainer.load_external_dataset(max_samples=max_samples)
        epochs = input("Neural training epochs (default 3): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 3
        trainer.train_comprehensive_model(neural_epochs=epochs)

    elif mode == '2':
        trainer = AdvancedTrainer(chatbot)
        trainer.interactive_training_session()

    elif mode == '3':
        print("\nEntering test chat mode. Type 'exit' to quit.")
        print("Commands: 'stats', 'sentiment <text>', 'personality <trait> <value>'")
        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'stats':
                    stats = chatbot.get_training_stats()
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.startswith('sentiment '):
                    text = user_input[10:]
                    result = chatbot.predict_sentiment(text)
                    print(f"Sentiment: {result}")
                    continue
                elif user_input.startswith('personality '):
                    parts = user_input.split()
                    if len(parts) == 3:
                        trait, value = parts[1], float(parts[2])
                        chatbot.set_personality({trait: value})
                        print(f"Set personality trait: {trait} = {value}")
                    continue

                use_neural = input("Use neural model? (y/n, default y): ").strip().lower()
                use_neural = use_neural != 'n'
                response = chatbot.chat(user_input, use_neural=use_neural)
                print(f"AI: {response}")

                sentiment = chatbot.predict_sentiment(user_input)
                if sentiment['sentiment'] != 'unknown':
                    print(f"Your sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.2f})")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    elif mode == '4':
        test_inputs = [
            "Hello, how are you?",
            "What's the weather like?",
            "Can you help me with programming?",
            "Tell me a joke",
            "What do you think about artificial intelligence?",
            "How do I cook pasta?",
            "What's the meaning of life?",
            "Can you explain quantum physics?",
            "What's your favorite color?",
            "How do I learn machine learning?"
        ]
        results = benchmark_model_performance(chatbot, test_inputs)

        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Benchmark results saved to 'benchmark_results.json'")

    elif mode == '5':
        stats = chatbot.get_training_stats()
        patterns = chatbot.analyze_conversation_patterns()

        print("\nModel Analysis:")
        print("=" * 20)
        print("Training Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nConversation Patterns:")
        for key, value in patterns.items():
            print(f"  {key}: {value}")

    else:
        print("Invalid mode selected!")
        return

    chatbot.save_model()
    print("\nSession completed. Model saved.")

if __name__ == "__main__":
    main()
