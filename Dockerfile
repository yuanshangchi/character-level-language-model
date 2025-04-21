FROM pytorch/pytorch:2.2.0-cpu
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm

COPY requirements.txt /job/ 
RUN pip install -r requirements.txt
