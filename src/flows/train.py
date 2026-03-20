import torch

from src.utils.loss_visualizer import update_plot
from src.data.data import decode_autoregressive


def train(config, dataloader, model, loss_fn, optimizer, scheduler, loss_history=None):
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        charge, premz, mz, i, src_seq, tgt_seq, tgt_padding_mask = batch
        prem = premz * charge
        
        logits, encoded, decoded = model(
            mz.to(config.device),
            i.to(config.device),
            premz.to(config.device),
            tgt_seq.to(config.device),
            tgt_padding_mask.to(config.device)
        )
        
        predicted = logits.argmax(dim=-1)
        pred_str = decode_autoregressive(predicted[0], config)
        tgt_str = decode_autoregressive(tgt_seq[0], config)
        print(f"Pred: {pred_str}")
        print(f"Tgt:  {tgt_str}")
        
        loss = loss_fn(logits, encoded, decoded, tgt_seq.to(config.device), tgt_padding_mask.to(config.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_value = loss.item()
        print(f"Loss: {loss_value}")

        # update_plot(loss_history, loss_value)


def evaluate(config, dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            charge, premz, mz, i, src_seq, tgt_seq, tgt_padding_mask = batch
            logits, encoded, decoded = model(
                mz.to(config.device),
                i.to(config.device),
                premz.to(config.device),
                tgt_seq.to(config.device),
                tgt_padding_mask.to(config.device)
            )
            loss = loss_fn(logits, encoded, decoded, tgt_seq.to(config.device), tgt_padding_mask.to(config.device))
            test_loss += loss.item()
            num_batches += 1
    test_loss /= num_batches if num_batches > 0 else 1
    print(f"Test loss: {test_loss}")


def generate(config, dataloader, model):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            charge, premz, mz, i, src_seq, tgt_seq, tgt_padding_mask = batch
            generated = model(
                mz.to(config.device),
                i.to(config.device),
                premz.to(config.device)
            )
            
            gen_str = decode_autoregressive(generated[0], config)
            tgt_str = decode_autoregressive(tgt_seq[0], config)
            print(f"Generated: {gen_str}")
            print(f"Target:    {tgt_str}")
            print()
            
            if idx >= 4:
                break


def run(config, model, loss_fn, optimizer, scheduler, train_dataloader, eval_dataloader):
    loss_history = []
    for epoch in range(config.hyper.epochs):
        print(f"Epoch: {epoch}")
        train(config, train_dataloader, model, loss_fn, optimizer, scheduler, loss_history)
        evaluate(config, eval_dataloader, model, loss_fn)
        torch.save(model.state_dict(), f"{config.hyper.checkpoint_name}")
