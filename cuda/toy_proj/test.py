import torch
import time

if __name__=="__main__":
    a = torch.rand([4096, 256], dtype=torch.float32).cuda()
    b = torch.rand([256, 1024], dtype=torch.float32).cuda()
    
    total = 0
    
    for i in range(10):
        torch.matmul(a, b)
        
    torch.cuda.synchronize()
    for i in range(1000):
        start = time.time()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        
        t = time.time() - start
        total += t
    
    print(total / 1000)
    
