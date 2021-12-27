import torch

if __name__ == "__main__":
    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
    else:
        print("no gpu")