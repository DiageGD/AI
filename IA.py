# ===================================================================
# Les imports (OS, bibliothèque CUDA, torch)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn



# ===================================================================
# Utilisation du GPU et vérification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.cuda.is_available())

torch.manual_seed(42)

if device.type == "cuda":
    cudnn.benchmark = True

torch.cuda.empty_cache()



# ===================================================================
# Utilisation des corpus et synthétisation du texte

files = [
    "corpus/used/corpus - art of poetry.txt",
    "corpus/used/corpus - witch of the demon seas.txt",
    "corpus/used/corpus - world of the drone.txt",
    "corpus/used/corpus - frankenstein.txt",
    "corpus/used/corpus - moby dick.txt",
    "corpus/used/corpus - romeo and juliett.txt",
    "corpus/used/corpus - pride and prejudice.txt",
    "corpus/used/corpus - jane eyre autobiography.txt",
    "corpus/used/corpus - alice in wonderland.txt",
    "corpus/used/corpus - city of god.txt",
    "corpus/used/corpus - picture of dorian gray.txt",
    "corpus/used/corpus - dracula.txt",
    "corpus/used/corpus - formula for conquest.txt"
]

# Note : Les corpus sont essentiellement en anglais
# pour être parler couramment, cette langue nécessite
# moins de mots que le français, c'était donc un choix optimal.

text = ""

for fname in files:
    with open(fname, "r", encoding="utf-8") as f:
        text += f.read() + "\n"

print("Nombre total de caractères :", len(text))



# ===================================================================
# Construction du vocabulaire

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    if isinstance(l, int):
        return itos[l]
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)



# ===================================================================
# Paramètres

hidden_size = 1024
block_size = 112
batch_size = 32
num_layers = 3
learning_rate = 3e-4
epochs = 10
steps = (len(text) // block_size) * epochs



# ===================================================================
# Modèle LSTM + Attention

class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.norm(out)

        attn_scores = self.attn(out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * out, dim=1)
        logits = self.fc(context)

        return logits, hidden


model = CharLSTM().to(device)



# ===================================================================
# Optimisation (cette partie était trop compliquée)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
scaler = torch.amp.GradScaler('cuda')



# ===================================================================
# Mise en place/utilisation du checkpoint

os.makedirs("saves", exist_ok=True)
checkpoint_path = "saves/checkpointVT.pt"
start_step = 0

if os.path.exists(checkpoint_path):
    print("Checkpoint trouvé. Reprise en cours...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_step = checkpoint['step'] + 1
    print(f"Reprise depuis step {start_step}")



# ===================================================================
# Entraînement (cette partie est la plus longue)
model.train()
for step in range(start_step, steps):

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x_batch = torch.stack([data[i:i+block_size] for i in ix])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    hidden = (
        torch.zeros(num_layers, batch_size, hidden_size, device=device),
        torch.zeros(num_layers, batch_size, hidden_size, device=device)
    )

    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        logits, hidden = model(x_batch, hidden)
        loss = loss_fn(logits, y_batch[:, -1])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if step % 100 == 0:
        print(f"{(step/steps)*100:.2f}% | Loss: {loss.item():.4f}")

    if step % 500 == 0 and step > 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, checkpoint_path)

        print(f"Checkpoint sauvegardé à step {step}")


# Sauvegarde finale
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
}, checkpoint_path)

print("Entraînement terminé. Checkpoint final sauvegardé.")



# ===================================================================
# Génération

model.eval()

temperature = 0.8
context = torch.randint(0, vocab_size, (1, 1)).to(device)

hidden = (
    torch.zeros(num_layers, 1, hidden_size, device=device),
    torch.zeros(num_layers, 1, hidden_size, device=device))

with torch.no_grad():
    for _ in range(310):

        with torch.amp.autocast('cuda'):
            logits, hidden = model(context, hidden)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        next_char = torch.multinomial(probs, 1)

        print(decode(next_char.item()), end="")

        context = next_char.view(1, 1)