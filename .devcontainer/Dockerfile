FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt update
RUN apt install -y git curl tmux wget
RUN apt install -y python3 python3-pip

# RUN pip install peft==0.7.0
# RUN pip install accelerate==0.21.0 tqdm==4.65.0 scipy==1.11.1
# RUN pip install transformers==4.35.2 lightning==2.0.5 datasets==2.13.1 matplotlib==3.8.0 tensorboardX==2.6.2.2

# RUN pip install bitsandbytes==0.41.3.post2

WORKDIR /workspace

# requirements.txtをコピー
COPY requirements.txt .

# pipを使用して依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# アプリケーションがリッスンするポートを指定（必要に応じて）
EXPOSE 1234
