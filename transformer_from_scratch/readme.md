# Transformer From Scratch in PyTorch 🧠⚡

![cover](assets/transformer_cover.png)

A simplified, readable implementation of the Transformer model, inspired by the paper _Attention is All You Need_, coded in PyTorch.

> 📚 Code credit: [Aladdin Persson's GitHub](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)

---

## 🚀 Key Concepts

### 1. **Multi-Head Self-Attention**
Self-attention lets the model weigh the importance of each token in a sequence relative to others.
Multiple heads allow it to capture different relationships in parallel.

```python
energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
```

**Explanation of `einsum`:**
- `nqhd`: batch (n), query length (q), heads (h), head_dim (d)
- `nkhd`: batch (n), key length (k), heads (h), head_dim (d)
- → Output shape: `nhqk` — attention scores per head between queries and keys.

Think of `einsum("nqhd,nkhd->nhqk")` as:
> For each head and batch, compute the dot product between each query and every key.

---

### 2. **Look-Ahead Mask (Causal Mask)**

Used in the decoder to prevent peeking at future tokens.

Imagine training the model to generate:
```
["I", "love", "deep", "learning"]
```

#### Mask Matrix:
```
         "I"  "love"  "deep"  "learning"
"I"        1     0       0        0
"love"     1     1       0        0
"deep"     1     1       1        0
"learning" 1     1       1        1
```

#### Step-by-step masking:

- **Step 0**: Predict `"I"` — sees only `"I"`
- **Step 1**: Predict `"love"` — sees `"I", "love"`
- **Step 2**: Predict `"deep"` — sees `"I", "love", "deep"`
- **Step 3**: Predict `"learning"` — sees all

✅ **Why**: Prevents cheating during training. Enforces autoregressive behavior.

---

### 3. **Padding Mask**
Used to ignore `<pad>` tokens during attention:

```python
src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
# → Shape: (batch, 1, 1, src_len)
```

Zeroes out attention for padded positions.

---

### 4. **Positional Embedding**
Adds position awareness to token embeddings:

```python
positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
out = word_embedding(x) + position_embedding(positions)
```

No recurrence → positional embedding is crucial.

---

## 🧩 Model Architecture

```
Input Embedding + Positional Embedding
        ↓
      Encoder
        ↓
Decoder ← Target Sequence + Look-Ahead Mask
        ↓
  Linear Layer → Output Tokens
```

---

## 🔍 Common Beginner Confusions

| Concept           | Clarification |
|------------------|--------------|
| `einsum`         | Just concise matrix multiplication. Easier than it looks. |
| Look-Ahead Mask  | Prevents the decoder from seeing future tokens. |
| Padding Mask     | Avoids attending to pad tokens. |
| Tensor Shapes    | Watch dimensions: batch first (N), then sequence length. |

---

## 💻 Key Line to Run

```python
out = model(src, trg[:, :-1])  # Shift target to predict next token
```

---

## 📐 Tensor Shape Summary

| Tensor          | Shape                        | Description |
|----------------|------------------------------|-------------|
| `queries`      | (N, query_len, heads, d)      | Queries per head |
| `energy`       | (N, heads, query_len, key_len)| Dot products between Q and K |
| `attention`    | (N, heads, query_len, key_len)| Normalized attention scores |
| `out`          | (N, query_len, embed_size)    | Final transformer output |

---

## ✅ Why Use This?

- Learn inner workings of attention
- Clear code for experimentation
- Minimal dependencies
- Great for interviews, education, and intuition-building

---

## 📷 Author & Credit

Created by [Aladdin Persson](https://www.youtube.com/c/AladdinPersson)  
Source GitHub: [Transformer From Scratch](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)

