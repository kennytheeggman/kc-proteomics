import torch

from src.utils.ctc import decode, reduce, decode_temporary


def train(config, dataloader, model, loss_fn, optimizer, scheduler):  # fix decode imputs: verify masses
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        charge, premz, mz, i, peptide = batch
        prem = premz*charge
        for idx in range(config.hyper.batch_size):
            prem_cur = prem[idx]
            prob_matrix, encoded, decoded = model(mz[idx].to(config.device), i[idx].to(config.device))
            print("".join(reduce(decode_temporary(prob_matrix).to(config.cpu))), peptide[idx].decode())
            # print("".join(reduce(decode(prob_matrix, config.aa_masses, prem_cur, config.tolerance, config.mass_res).to(config.cpu))), peptide[idx].decode())
            loss = loss_fn(prob_matrix, encoded, decoded, peptide[idx].decode())
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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
            for idx in range(config.hyper.batch_size):
                prob_matrix, encoded, decoded = model(mz[0].to(config.device), i[0].to(config.device))
                loss = loss_fn(prob_matrix, encoded, decoded, peptide)
                test_loss += loss.item()
    test_loss /= len(dataloader) * config.hyper.batch_size
    print(f"Test loss: {test_loss}")

def run(config, model, loss_fn, optimizer, scheduler, train_dataloader, eval_dataloader):
    for epoch in range(config.hyper.epochs):
        print(f"Epoch: {epoch}")
        train(config, train_dataloader, model, loss_fn, optimizer, scheduler)
        evaluate(config, eval_dataloader, model, loss_fn)
        torch.save(model.state_dict(), f"{config.hyper.checkpoint_name}")
