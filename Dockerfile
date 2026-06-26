FROM mambaorg/micromamba:1.5.10

LABEL maintainer="QiangSu"
LABEL description="scBOA: single-cell Bayesian Optimization and Analysis"

WORKDIR /workspace

COPY environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . /workspace

ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

CMD ["python", "scBOA.py", "--help"]