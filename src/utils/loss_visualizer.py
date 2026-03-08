import plotext as plt

def update_plot(loss_history, loss_val):
    loss_history.append(loss_val)
    plt.clf()
    plt.plot(list(loss_history))  
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plotsize(100, 30) 
    plt.show()