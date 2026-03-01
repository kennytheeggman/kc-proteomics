import torch

from src.utils.ctc import decode, reduce


def train(config, dataloader, model, loss_fn, optimizer):
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        charge, premz, mz, i, peptide = batch
        for idx in range(config.hyper.batch_size):
            prob_matrix, encoded, decoded = model(mz[idx].to(config.device), i[idx].to(config.device))
            print("".join(reduce(decode(prob_matrix, log=True).to(config.cpu))), peptide[idx].decode())
            loss = loss_fn(prob_matrix, encoded, decoded, peptide[idx].decode())
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
        # # 3. Print gradients for all parameters
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}") # Using norm for cleaner output
        #         # print(param.grad) # Print raw tensor
        #     else:
        #         print(f"No gradient for {name}")

def evaluate(config, dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            charge, premz, mz, i, peptide = batch
            prob_matrix, encoded, decoded = model(mz[0].to(config.device), i[0].to(config.device))
            loss = loss_fn(prob_matrix, encoded, decoded, peptide)
            test_loss += loss.item()
    test_loss /= len(dataloader)
    print(f"Test loss: {test_loss}")

def run(config, model, loss_fn, optimizer, train_dataloader, eval_dataloader):
    for epoch in range(config.hyper.epochs):
        print(f"Epoch: {epoch}")
        train(config, train_dataloader, model, loss_fn, optimizer)
        evaluate(config, eval_dataloader, model, loss_fn)
        torch.save(model.state_dict(), f"{config.hyper.checkpoint_name}")
