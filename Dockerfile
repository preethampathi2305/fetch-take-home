FROM amazonlinux:2023

WORKDIR /app

COPY . /app

RUN yum install python39

RUN yum install -y python3-pip

RUN yum install -y wget

RUN wget https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=86cc28df491fa84738affe752f9870791026565342f69e4ab63e5b935f00a495

RUN pip install torch-2.1.0+cpu-cp39-cp39-linux_x86_64.whl

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "search.py"]