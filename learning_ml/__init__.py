import hw3
import time
import wandb

# start_time = time.time()
wandb.init(
    # set the wandb project where this run will be logged
    project="pytorch_hw3",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.016,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 256,
    }
)
hw3.train(epochs=256, wandb=wandb)
hw3.test()
wandb.finish()
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# hw3.train(epochs=16, cuda=False)
# hw3.test(cuda=False)
# print("--- %s seconds ---" % (time.time() - start_time))

# hw3.train(epochs=4)
# hw3.test()