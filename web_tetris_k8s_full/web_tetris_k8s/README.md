
# Web Tetris (K8s-ready, PVC leaderboard)

Single-player Tetris chạy trên **browser**, server **FastAPI** serve static web và API lưu điểm vào **SQLite** trên PVC.
Expose qua **NodePort** để mọi người cùng truy cập.

## Thư mục
```
server/
  ├── server.py
  ├── requirements.txt
  ├── Dockerfile
  └── static/
      ├── index.html
      ├── leaderboard.html
      ├── game.js
      └── style.css
k8s/
  ├── pvc.yaml
  └── server.yaml
```

## Build & Push image
```bash
cd server
docker build -t YOUR_DOCKER/web-tetris:latest .
docker push YOUR_DOCKER/web-tetris:latest
```

## Deploy lên Kubernetes (NodePort + PVC)
```bash
kubectl apply -f k8s/pvc.yaml
# Sửa image trong k8s/server.yaml cho đúng repo của bạn
kubectl apply -f k8s/server.yaml
```

## Truy cập
- Game: `http://<node-ip>:30080/`
- Leaderboard: `http://<node-ip>:30080/leaderboard`
- Health: `http://<node-ip>:30080/healthz`

## Reset leaderboard
Xoá file DB trong PVC (hoặc xóa PVC và apply lại).
