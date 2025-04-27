#!/bin/bash
# /usr/local/bin/your_program_monitor.sh

MAX_RETRIES=5  # 最大重试次数
RETRY_DELAY=10 # 每次重试间隔(秒)
PROGRAM_PATH="/path/to/tracking_code.py" # 替换为你的 tracking_code.py 实际路径
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