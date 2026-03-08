
#### 启动训练
1. 将训练过程输出到 train.log，即使断开 SSH 也会继续训练
    ```
    nohup python train.py > train.log 2>&1 &
    ```
2. 实时查看训练进度
    ```
    tail -f train.log
    ```
