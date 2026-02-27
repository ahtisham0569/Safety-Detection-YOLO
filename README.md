Model file (best.pt) not included due to size limitations.
## Training

python train.py

## Deployment

docker build -t safety-app .
docker run -p 5000:5000 safety-app
