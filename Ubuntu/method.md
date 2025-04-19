# Linux 开机自启动脚本（带错误监控和系统重启功能）

下面是一个完整的解决方案，包含一个 systemd 服务和一个监控脚本，当程序崩溃时会自动重启，如果连续失败超过指定次数则重启整个系统。

## 方案一：使用 systemd 服务 + Bash 监控脚本

### 1. 创建监控脚本

```bash
#!/bin/bash
# /usr/local/bin/your_program_monitor.sh

MAX_RETRIES=5  # 最大重试次数
RETRY_DELAY=10 # 每次重试间隔(秒)
PROGRAM_PATH="/path/to/your/program" # 替换为你的程序路径
PROGRAM_ARGS="" # 程序参数(如果有)
LOG_FILE="/var/log/your_program.log" # 日志文件路径

# 计数器
FAIL_COUNT=0

while true; do
    # 运行程序并记录输出
    echo "$(date) - 启动程序: $PROGRAM_PATH $PROGRAM_ARGS" >> "$LOG_FILE"
    $PROGRAM_PATH $PROGRAM_ARGS >> "$LOG_FILE" 2>&1
    
    # 检查程序退出状态
    EXIT_STATUS=$?
    echo "$(date) - 程序退出，状态码: $EXIT_STATUS" >> "$LOG_FILE"
    
    # 如果程序正常退出(0)，则重置计数器
    if [ $EXIT_STATUS -eq 0 ]; then
        FAIL_COUNT=0
        echo "$(date) - 程序正常退出，停止监控" >> "$LOG_FILE"
        exit 0
    fi
    
    # 增加失败计数器
    ((FAIL_COUNT++))
    
    # 检查是否达到最大重试次数
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "$(date) - 错误: 程序连续失败 $MAX_RETRIES 次，将重启系统" >> "$LOG_FILE"
        # 写入系统日志
        logger -t your_program_monitor "程序连续失败 $MAX_RETRIES 次，即将重启系统"
        # 延迟10秒后重启
        sleep 10
        reboot
        exit 1
    fi
    
    # 等待后重试
    echo "$(date) - 将在 $RETRY_DELAY 秒后重试 (尝试 $FAIL_COUNT/$MAX_RETRIES)" >> "$LOG_FILE"
    sleep $RETRY_DELAY
done
```

给脚本添加执行权限：
```bash
sudo chmod +x /usr/local/bin/your_program_monitor.sh
```

### 2. 创建 systemd 服务文件

```bash
# /etc/systemd/system/your_program.service
[Unit]
Description=Your Program with Auto-Restart
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/your_program_monitor.sh
Restart=no  # 我们自己在脚本中处理重启逻辑
User=yourusername  # 替换为运行程序的用户
WorkingDirectory=/path/to/your/program/directory  # 替换为程序工作目录

[Install]
WantedBy=multi-user.target
```

### 3. 启用并启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable your_program.service
sudo systemctl start your_program.service
```

## 方案二：纯 systemd 实现（更简单）

如果你更喜欢只使用 systemd 的功能而不需要额外脚本：

```bash
# /etc/systemd/system/your_program.service
[Unit]
Description=Your Program with Auto-Restart
After=network.target

[Service]
Type=simple
ExecStart=/path/to/your/program
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=5
User=yourusername
WorkingDirectory=/path/to/your/program/directory

# 当连续5次在60秒内启动失败时，执行系统重启
StartLimitAction=reboot-force

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable your_program.service
sudo systemctl start your_program.service
```

## 日志查看

查看程序日志：
```bash
journalctl -u your_program.service -f
```

或查看监控脚本的日志：
```bash
tail -f /var/log/your_program.log
```

## 注意事项

1. 请确保替换所有 `/path/to/your/program` 为实际程序路径
2. 根据程序特性调整 `MAX_RETRIES` 和 `RETRY_DELAY` 参数
3. 对于生产环境，建议先在测试系统上验证此配置
4. 频繁系统重启可能导致问题，请谨慎设置重试次数
5. 如果程序是GUI应用，可能需要设置 `DISPLAY` 环境变量

这个方案提供了完整的错误处理和系统重启功能，同时保留了详细的日志记录以便问题排查。