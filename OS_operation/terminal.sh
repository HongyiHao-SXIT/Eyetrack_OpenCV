# 提权
sudo chmod +x /usr/local/bin/your_program_monitor.sh

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable your_program.service
sudo systemctl start your_program.service