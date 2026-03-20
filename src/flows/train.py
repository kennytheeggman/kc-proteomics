import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.ctc import decode, reduce, decode_temporary
from src.utils.loss_visualizer import update_plot

def train(config, dataloader, model, loss_fn, optimizer, scheduler, writer, global_step, loss_history=None):  # fix decode imputs: verify masses
    model.train()
    moving_avg = None
    weight = 0.1
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        charge, premz, mz, i, peptide = batch
        prem = premz*charge
        
        prob_matrix, encoded, decoded = model(mz.to(config.device), i.to(config.device), premz.to(config.device))
        print("".join(reduce(decode_temporary(prob_matrix[0]).to(config.cpu))), peptide[0].decode())
        # print("".join(reduce(decode(prob_matrix[0], config.aa_masses, prem[0], config.tolerance, config.mass_res).to(config.cpu))), peptide[0].decode())
        loss = loss_fn(prob_matrix, encoded, decoded, peptide)  # !!!
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_value = loss.item()
        print(f"Loss: {loss_value}")

        # temporary visualizer, comment out if we dont need
        update_plot(loss_history, loss_value)

        # logging loss, learning rate, moving avg
        if moving_avg is None:
            moving_avg = loss_value
        else:
            moving_avg = loss_value * weight + (1 - weight) * moving_avg

        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("Moving Average Loss", moving_avg, global_step)
        global_step += 1

    return global_step

        # # 3. Print gradients for all parameters
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}") # Using norm for cleaner output
        #         # print(param.grad) # Print raw tensor
        #     else:
        #         print(f"No gradient for {name}")

def evaluate(config, dataloader, model, loss_fn, writer, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            charge, premz, mz, i, peptide = batch
            prob_matrix, encoded, decoded = model(mz.to(config.device), i.to(config.device), premz.to(config.device))
            loss = loss_fn(prob_matrix, encoded, decoded, peptide)
            if loss is None:
                continue
            test_loss += loss.item()
    test_loss /= len(dataloader)
    print(f"Test loss: {test_loss}")

def run(config, model, loss_fn, optimizer, scheduler, train_dataloader, eval_dataloader):
    loss_history = []
    writer = SummaryWriter(log_dir="runs/run1")
    global_step = 0
    for epoch in range(config.hyper.epochs):
        print(f"Epoch: {epoch}")
        global_step = train(config, train_dataloader, model, loss_fn, optimizer, scheduler, writer, global_step, loss_history)
        evaluate(config, eval_dataloader, model, loss_fn, writer, epoch)
        torch.save(model.state_dict(), f"{config.hyper.checkpoint_name}")
    writer.close()
