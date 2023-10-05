from pathlib import Path
import torch
import torch.nn as nn


from src.config import config
from src.models.transformers import Transformer
from src.utils import WandbLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.load(config.pytorch_dir  / "X_train_all.pt")

y = torch.load(config.pytorch_dir / "y_train_all.pt").to(device)
X_test = torch.load(config.pytorch_dir / "X_test_all.pt")[:1000].to(device)
y_test = torch.load(config.pytorch_dir / "y_test_all.pt")[:1000].to(device)
wanddblogger = WandbLogger(project_name="crypto", entity_name="pab_lo4", model_params=config.dict())

print(config.dict())

m = Transformer(config).to(device)
if config.load_model:
    m.load_state_dict(torch.load(config.weights_dir / "weights.pt"))
    print("Model loaded from", config.weights_dir / "weights.pt")

#BS = 10
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

loss_training_l = []
loss_val_l = []

for i in range(config.epochs):
    idx=torch.randint(0, X.size()[0], (config.batch_size,))
    #m.zero_grad()
    optimizer.zero_grad()
    out = m(X[idx].squeeze().to(device))
    # some broadcasting magic (from mingpt)
    loss = loss_fn(out.view(config.batch_size*config.block_size,config.vocab_size),y[idx].view(config.batch_size*config.block_size))
    loss.backward()
    optimizer.step()
    if i % config.evaluation_steps==0:
        loss_training = loss.item()
        loss_training_l.append(loss_training)
        with torch.no_grad():
            m=m.eval()
            out = m(X_test.squeeze())
            loss = loss_fn(out.view(-1,config.vocab_size),y_test.view(-1))
            loss_val = loss.item()
            loss_val_l.append(loss_val)
            m.train()
            print("i:", i, "Loss train", loss_training, "Loss val", loss_val)
            torch.save(m.state_dict(), config.weights_dir / "weights.pt")
            wanddblogger.log({"loss_train":loss_training, "loss_val":loss_val, "epoch":i})



# save model
torch.save(m.state_dict(), config.weights_dir / "weights.pt")
wanddblogger.log_weights(m.state_dict(), name="weights")
# save config as json
# import json
# with open(config.weights_dir  / "config.json", "w") as f:
#     json.dump(config.json(), f)

# save losses
import matplotlib.pyplot as plt
plt.plot(loss_training_l, label="training")
plt.plot(loss_val_l, label="validation")
plt.legend()
plt.savefig(config.weights_dir / "loss.png")